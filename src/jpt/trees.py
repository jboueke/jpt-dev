'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import html
import json
import queue
from operator import attrgetter
from threading import Lock

import math
import numbers
import operator
import os
import pickle
import pprint
from collections import defaultdict, deque, ChainMap, OrderedDict
import datetime
from itertools import zip_longest
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import numpy.lib.stride_tricks
import pandas as pd
from graphviz import Digraph
from matplotlib import style, pyplot as plt

import dnutils
from dnutils import first, ifnone, mapstr, err, fst

import jpt.variables
from .base.utils import Unsatisfiability

from .variables import VariableMap, SymbolicVariable, NumericVariable, Variable, ScaledNumeric
from .learning.distributions import Distribution

from .base.utils import list2interval, format_path, normalized
from .learning.distributions import Multinomial, Numeric, SymbolicType
from .base.constants import plotstyle, orange, green, SYMBOL

try:
    from .base.quantiles import __module__
    from .base.intervals import __module__
    from .learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.quantiles import QuantileDistribution, LinearFunction
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet
    from .learning.impurity import Impurity

style.use(plotstyle)

# ----------------------------------------------------------------------------------------------------------------------
# Global data store to exploit copy-on-write in multiprocessing

import multiprocessing as mp

_data = None
_lock = Lock()


# ----------------------------------------------------------------------------------------------------------------------


def _prior(args):
    var_idx, json_var = args
    try:
        return Variable.from_json(json_var).dist(data=_data, col=var_idx).to_json()
    except ValueError as e:
        raise ValueError('%s: %s' % (Variable.from_json(json_var), str(e)))


# ----------------------------------------------------------------------------------------------------------------------
# Global constants

DISCRIMINATIVE = 'discriminative'
GENERATIVE = 'generative'


# ----------------------------------------------------------------------------------------------------------------------


class Node:
    '''
    Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx: int, parent: Union[None, 'DecisionNode'] = None) -> None:
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        '''
        self.idx = idx
        self.parent: DecisionNode = parent
        self.samples = 0.
        self._path = []

    @property
    def path(self) -> VariableMap:
        res = VariableMap()
        for var, vals in self._path:
            res[var] = res.get(var, set(range(var.domain.n_values)) if var.symbolic else R).intersection(vals)
        return res

    def consistent_with(self, evidence: VariableMap) -> bool:
        """
        Check if the node is consistent with the variable assignments in evidence.

        :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :type evidence: VariableMap
        """

        # for every variable and its assignment
        for variable, value in evidence.items():
            variable: Variable

            # if the variable is in the path of this node
            if variable in self.path.keys():

                # get the restriction of the path
                restriction = self.path[variable]

                # if it is a numeric
                if variable.numeric:

                    # and a range is given
                    if isinstance(value, ContinuousSet):
                        # if the ranges don't intersect return false
                        if value.intersection(restriction).isempty():
                            return False

                    # if it is a singular value
                    else:
                        # check if the path allows this value
                        if not restriction.lower < value <= restriction.upper:
                            return False

                # if the variable is symbolic
                elif variable.symbolic:

                    # if it is a set of possible values
                    if isinstance(restriction, set):

                        # check if the sets intersect
                        if len(restriction & value) == 0:
                            return False

                    # if it is a singular observation
                    else:

                        # check if the path allows this value
                        if value not in restriction:
                            return False

        return True

    def format_path(self):
        return format_path(self.path)

    def __str__(self) -> str:
        return f'Node<{self.idx}>'

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'

    def depth(self):
        return len(self._path)

    def contains(self, samples: np.ndarray, variable_index_map: VariableMap) -> np.array:
        """ Check if this node contains the given samples in parallel.

        @param samples: The samples to check
        @param variable_index_map: A VariableMap mapping to the indices in 'samples'
        @return numpy array with 0s and 1s
        """
        result = np.ones(len(samples))
        for variable, restriction in self.path.items():
            index = variable_index_map[variable]
            if variable.numeric:
                result *= (samples[:, index] > restriction.lower) & (samples[:, index] <= restriction.upper)
            if variable.symbolic:
                result *= np.isin(samples[:, index], list(restriction))

        return result

# ----------------------------------------------------------------------------------------------------------------------


class DecisionNode(Node):
    '''
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx: int, variable: Variable, parent: 'DecisionNode' = None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param variable:   the split feature name
        :type variable:    jpt.variables.Variable
        '''
        self._splits = None
        self.variable = variable
        super().__init__(idx, parent=parent)
        self.children: None or List[Node] = None  # [None] * len(self.splits)

    def __eq__(self, o) -> bool:
        return (type(self) is type(o) and
                self.idx == o.idx and
                (self.parent.idx
                 if self.parent is not None else None) == (o.parent.idx if o.parent is not None else None) and
                [n.idx for n in self.children] == [n.idx for n in o.children] and
                self.splits == o.splits and
                self.variable == o.variable and
                self.samples == o.samples)

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'parent': ifnone(self.parent, None, attrgetter('idx')),
                'splits': [s.to_json() if isinstance(s, ContinuousSet) else list(s) for s in self.splits],
                'variable': self.variable.name,
                '_path': [(var.name, split.to_json() if var.numeric else list(split)) for var, split in self._path],
                'children': [node.idx for node in self.children],
                'samples': self.samples,
                'child_idx': self.parent.children.index(self) if self.parent is not None else None}

    @staticmethod
    def from_json(jpt: 'JPT', data: Dict[str, Any]) -> 'DecisionNode':
        node = DecisionNode(idx=data['idx'], variable=jpt.varnames[data['variable']])
        node.splits = [Interval.from_json(s) if node.variable.numeric else set(s) for s in data['splits']]
        node.children = [None] * len(node.splits)
        node.parent = ifnone(data['parent'], None, jpt.innernodes.get)
        node.samples = data['samples']
        if node.parent is not None:
            node.parent.set_child(data['child_idx'], node)
        jpt.innernodes[node.idx] = node
        return node

    @property
    def splits(self) -> List:
        return self._splits

    @splits.setter
    def splits(self, splits):
        if self.children is not None:
            raise ValueError('Children already set: %s' % self.children)
        self._splits = splits
        self.children = [None] * len(self._splits)

    def set_child(self, idx: int, node: Node) -> None:
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.variable, self.splits[idx]))

    def str_edge(self, idx) -> str:
        if self.variable.numeric:
            return str(ContinuousSet(self.variable.domain.labels[self.splits[idx].lower],
                                     self.variable.domain.labels[self.splits[idx].upper],
                                     self.splits[idx].left,
                                     self.splits[idx].right))
        else:
            negate = len(self.splits[1]) > 1
            if negate:
                label = self.variable.domain.labels[fst(self.splits[0])]
                return '%s%s' % ('\u00AC' if idx > 0 else '', label)
            else:
                return str(self.variable.domain.labels[fst(self.splits[idx])])

    @property
    def str_node(self) -> str:
        return self.variable.name

    def recursive_children(self):
        return self.children + [item for sublist in
                                [child.recursive_children() for child in self.children] for item in sublist]

    def __str__(self) -> str:
        return (f'<DecisionNode #{self.idx} '
                f'{self.variable.name} = [%s]' % '; '.join(self.str_edge(i) for i in range(len(self.splits))) +
                f'; parent-#: {self.parent.idx if self.parent is not None else None}'
                f'; #children: {len(self.children)}>')

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class Leaf(Node):
    '''
    Represents a leaf node of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx: int, parent: Node or None = None, prior=None):
        super().__init__(idx, parent=parent)
        self.distributions = VariableMap()
        self.prior = prior

    @property
    def str_node(self) -> str:
        return ""

    def applies(self, query: VariableMap) -> bool:
        '''Checks whether this leaf is consistent with the given ``query``.'''
        path = self.path
        for var in set(query.keys()).intersection(set(path.keys())):
            if path.get(var).isdisjoint(query.get(var)):
                return False
        return True

    @property
    def value(self):
        return self.distributions

    def recursive_children(self):
        return []

    def __str__(self) -> str:
        return (f'<Leaf # {self.idx}; '
                f'parent: <%s # %s>>' % (type(self.parent).__qualname__, ifnone(self.parent, None, attrgetter('idx'))))

    def __repr__(self) -> str:
        return f'Leaf<{self.idx}> object at {hex(id(self))}'

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'distributions': self.distributions.to_json(),
                'prior': self.prior,
                'samples': self.samples,
                'parent': ifnone(self.parent, None, attrgetter('idx')),
                'child_idx': self.parent.children.index(self) if self.parent is not None else -1}

    @staticmethod
    def from_json(tree: 'JPT', data: Dict[str, Any]) -> 'Leaf':
        leaf = Leaf(idx=data['idx'], prior=data['prior'], parent=tree.innernodes.get(data['parent']))
        leaf.distributions = VariableMap.from_json(tree.variables, data['distributions'], Distribution)
        leaf._path = []
        if leaf.parent is not None:
            leaf.parent.set_child(data['child_idx'], leaf)
        leaf.prior = data['prior']
        leaf.samples = data['samples']
        tree.leaves[leaf.idx] = leaf
        return leaf

    def __eq__(self, o) -> bool:
        return (type(o) == type(self) and
                self.idx == o.idx and
                self._path == o._path and
                self.samples == o.samples and
                self.distributions == o.distributions and
                self.prior == o.prior)

    def consistent_with(self, evidence: VariableMap) -> bool:
        """
        Check if the node is consistent with the variable assignments in evidence.

        :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :type evidence: VariableMap
        """
        return self.probability(evidence) > 0.
    
    def path_consistent_with(self, evidence: VariableMap) -> bool:
        return super(Leaf, self).consistent_with(evidence)
    
    def probability(self, query: VariableMap, dirac_scaling: float = 2.,  min_distances: VariableMap = None) -> float:
        """
        Calculate the probability of a (partial) query. Exploits the independence assumption
        :param query: A VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :type query: VariableMap
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :type dirac_scaling: float
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :type min_distances: A VariableMap from numeric variables to floats or None
        """
        result = 1.
        # for every variable and its assignment
        for variable, value in query.items():
            variable: Variable

            # if it is a numeric
            if variable.numeric:
                # and a range is given
                if isinstance(value, ContinuousSet):
                    # multiply by probability which is possible due to independence
                    result *= self.distributions[variable]._p(value)

                # if it is a singular value
                else:
                    # get the likelihood
                    print(self.distributions[variable].pdf(value))
                    likelihood = self.distributions[variable].pdf(value)

                    # if it is infinity and no handling is provided replace it with 1.
                    if likelihood == float("inf") and not min_distances:
                        result *= 1
                    # if it is infinite and a handling is provided, replace with dirac_sclaing/min_distance
                    elif likelihood == float("inf") and min_distances:
                        result *= dirac_scaling / min_distances[variable]
                    else:
                        result *= likelihood

            # if the variable is symbolic
            elif variable.symbolic:

                # force the evidence to be a set
                if not isinstance(value, set):
                    value = set([value])

                # return false if the evidence is impossible in this leaf
                result *= self.distributions[variable].p(value)

        return result

    def parallel_likelihood(self, queries: np.ndarray, dirac_scaling: float = 2.,  min_distances: VariableMap = None) \
            -> float:
        """
        Calculate the probability of a (partial) query. Exploits the independence assumption
        :param queries: A VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :type queries: VariableMap
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :type dirac_scaling: float
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :type min_distances: A VariableMap from numeric variables to floats or None
        """

        # create result vector
        result = np.ones(len(queries))

        # for each idx, variable and distribution
        for idx, (variable, distribution) in enumerate(self.distributions.items()):

            # if the variable is symbolic
            if isinstance(variable, SymbolicVariable):

                # multiply by probability
                probs = distribution._params[queries[:, idx].astype(int)]

            # if the variable is numeric
            elif isinstance(variable, NumericVariable):

                # get the likelihoods
                probs = np.asarray(distribution.pdf.multi_eval(queries[:, idx].copy(order='C').astype(float)))

                if min_distances:
                    # replace them with dirac scaling if they are infinite
                    probs[(probs == float("inf")).nonzero()] = dirac_scaling / min_distances[variable]

                # if no distances are provided replace infinite values with 1.
                else:
                    probs[(probs == float("inf")).nonzero()] = 1.

            # multiply results
            result *= probs

        return result
# ----------------------------------------------------------------------------------------------------------------------


class Result:

    def __init__(self, query, evidence, res=None, cand=None, w=None):
        self.query = query
        self._evidence = evidence
        self._res = ifnone(res, [])
        self._cand = ifnone(cand, [])
        self._w = ifnone(w, [])

    def __str__(self):
        return self.format_result()

    @property
    def evidence(self):
        return {k: (k.domain.labels[fst(v)]
                    if k.symbolic else ContinuousSet(k.domain.labels[v.lower],
                                                     k.domain.labels[v.upper], v.left, v.right))
                for k, v in self._evidence.items()}

    @property
    def result(self):
        return self._res

    @result.setter
    def result(self, res):
        self._res = res

    @property
    def candidates(self):
        return self._cand

    @candidates.setter
    def candidates(self, cand):
        self._cand = cand

    @property
    def weights(self):
        return self._w

    @weights.setter
    def weights(self, w):
        self._w = w

    def format_result(self):
        return ('P(%s%s) = %.3f%%' % (format_path(self.query),
                                      (' | %s' % format_path(self.evidence)) if self.evidence else '',
                                      self.result * 100))

    def explain(self):
        result = self.format_result()
        result += '\n'
        for weight, leaf in sorted(zip(self.weights, self.candidates), key=operator.itemgetter(0), reverse=True):
            result += '%.3f%%: %s\n' % (weight,
                                        format_path({var: val for var, val in leaf.path.items()
                                                     if var not in self.evidence}))
        return result


class ExpectationResult(Result):

    def __init__(self, query, evidence, theta, lower=None, upper=None, res=None, cand=None, w=None):
        super().__init__(query, evidence, res=res, cand=cand, w=w)
        self.theta = theta
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self.query.domain.labels[self._lower]

    @property
    def upper(self):
        return self.query.domain.labels[self._upper]

    @property
    def result(self):
        return self.query.domain.labels[self._res]

    def format_result(self):
        left = 'E(%s%s%s; %s = %.3f)' % (self.query.name,
                                         ' | ' if self.evidence else '',
                                         # ', '.join([var.str(val, fmt='logic') for var, val in self._evidence.items()]),
                                         format_path(self.evidence),
                                         SYMBOL.THETA,
                                         self.theta)
        right = '[%.3f %s %.3f %s %.3f]' % (self.lower,
                                            SYMBOL.ARROW_BAR_LEFT,
                                            self.result,
                                            SYMBOL.ARROW_BAR_RIGHT,
                                            self.upper) if self.query.numeric else self.result
        return '%s = %s' % (left, right)


class MPEResult(Result):

    def __init__(self, evidence, res=None, cand=None, w=None):
        super().__init__(None, evidence, res=res, cand=cand, w=w)
        self.path = {}

    def format_result(self):
        return f'MPE({self.evidence}) = {format_path(self.path)}'


class PosteriorResult(Result):

    def __init__(self, query, evidence, dists=None, cand=None, w=None):
        super().__init__(query, evidence, res=None, cand=cand)
        self._w = ifnone(w, {})
        self.distributions: Dict[Variable, Distribution] = dists

    def format_result(self):
        return ('P(%s%s%s) = %.3f%%' % (', '.join([var.str(val, fmt="logic") for var, val in self.query.items()]),
                                        ' | ' if self.evidence else '',
                                        ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
                                        self.result * 100))

    def __getitem__(self, item):
        return self.distributions[item]

    def impurity(self, variables: None or List[Variable] = None):
        """Calculate the impurity (sum over variances and ginis) of the result of this query for the given variables.
        """
        # use all variables if none are given

        if variables is None:
            variables = list(self.distributions.keys())

        # initialize result
        result = 0.

        # for every requested variable
        for variable in variables:

            # get the distribution
            distribution = self.distributions[variable]

            # add variance if numeric
            if variable.numeric:
                result += distribution.variance()

            # add gini impurity if symbolic
            elif variable.symbolic:
                result += distribution.gini_impurity()

        return result

    def __eq__(self, other):
        if not isinstance(other, PosteriorResult):
            return False
        return self.result == other.result and self.distributions == other.distributions

class JPT:
    '''
    Joint Probability Trees.
    '''

    logger = dnutils.getlogger('/jpt', level=dnutils.INFO)

    def __init__(self, variables, targets=None, min_samples_leaf=.01, min_impurity_improvement=None,
                 max_leaves=None, max_depth=None, variable_dependencies=None) -> None:
        '''Implementation of Joint Probability Tree (JPT) learning. We store multiple distributions
        induced by its training samples in the nodes so we can later make statements
        about the confidence of the prediction.
        has children :class:`~jpt.learning.trees.Node`.

        :param variables:           the variable declarations of the data being processed by this tree
        :type variables:            [jpt.variables.Variable]
        :param min_samples_leaf:    the minimum number of samples required to generate a leaf node
        :type min_samples_leaf:     int or float
        :param variable_dependencies: A dict that maps every variable to a list of variables that are 
                                        directly dependent to that variable.
        :type variable_dependencies: None or Dict from variable to list of variables 
        '''

        self._variables = tuple(variables)
        self._targets = targets
        self.varnames: OrderedDict[str, Variable] = OrderedDict((var.name, var) for var in self._variables)
        self.leaves: Dict[int, Leaf] = {}
        self.innernodes: Dict[int, DecisionNode] = {}
        self.allnodes: ChainMap[int, Node] = ChainMap(self.innernodes, self.leaves)
        self.priors = {}

        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = ifnone(min_impurity_improvement, 0)
        self._numsamples = 0
        self.root = None
        self.c45queue = deque()
        self.max_leaves = max_leaves
        self.max_depth = max_depth or float('inf')
        self._node_counter = 0
        self.indices = None
        self.impurity = None

        # initialize the dependencies as fully dependent on each other.
        # the interface isn't modified therefore the jpt should work as before if not
        # specified different
        if variable_dependencies is None:
            self.variable_dependencies: VariableMap[Variable, List[Variable]] = \
                VariableMap(zip(self.variables, [list(self.variables)] * len(self.variables)))
        else:
            self.variable_dependencies: VariableMap[Variable, List[Variable]] = variable_dependencies

        # also initialize the dependency structure as indices since it will be usefull in the c45 algorithm
        self.dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                         -1, dtype=np.int64)

        # dependencies to numeric variables for every variable
        self.numeric_dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                                 -1,
                                                 dtype=np.int64)

        # dependencies to symbolic variables for every variable
        self.symbolic_dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                                  -1,
                                                  dtype=np.int64)

        # convert variable dependency structure to index dependency structure for easy interpretation in cython
        for key, value in self.variable_dependencies.items():

            # get the index version of the dependent variables and store them
            key_ = self.variables.index(key)
            value_ = [self.variables.index(var) for var in value]
            self.dependency_matrix[key_, 0:len(value_)] = value_

            # create lists to store the index dependencies for only numeric/symbolic variables
            numeric_dependencies = []
            symbolic_dependencies = []

            for dependent_variable in value:
                # skip dependent variables if one is not allowed to purify them
                if self.targets and dependent_variable not in self.targets:
                    continue

                # get index of numeric dependent variable
                if isinstance(dependent_variable, NumericVariable):
                    if self.targets:
                        numeric_dependencies.append(
                            self.numeric_targets.index(dependent_variable)
                        )
                    else:
                        numeric_dependencies.append(
                            self.numeric_variables.index(dependent_variable)
                        )

                # get indices of symbolic dependent variable
                elif isinstance(dependent_variable, SymbolicVariable):
                    if self.targets:
                        symbolic_dependencies.append(
                            self.symbolic_targets.index(dependent_variable)
                        )
                    else:
                        symbolic_dependencies.append(
                            self.symbolic_variables.index(dependent_variable)
                        )

            # save the index dependencies to the matrix later used to calculate impurities
            self.numeric_dependency_matrix[key_, 0:len(numeric_dependencies)] = numeric_dependencies
            self.symbolic_dependency_matrix[key_, 0:len(symbolic_dependencies)] = symbolic_dependencies

    def _reset(self) -> None:
        self.innernodes.clear()
        self.leaves.clear()
        self.priors.clear()
        self.root = None
        self.c45queue.clear()

    @property
    def variables(self) -> Tuple[Variable]:
        return self._variables

    @property
    def targets(self) -> List[Variable]:
        return self._targets

    @property
    def features(self) -> List[Variable]:
        return [var for var in self.variables if var not in self.targets]

    @property
    def numeric_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, NumericVariable)]

    @property
    def symbolic_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, SymbolicVariable)]

    @property
    def numeric_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, NumericVariable)]

    @property
    def symbolic_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, SymbolicVariable)]

    @property
    def numeric_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, NumericVariable)]

    @property
    def symbolic_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, SymbolicVariable)]

    def to_json(self) -> Dict:
        return {'variables': [v.to_json() for v in self.variables],
                'targets': [v.name for v in self.targets] if self.targets else self.targets,
                'min_samples_leaf': self.min_samples_leaf,
                'min_impurity_improvement': self.min_impurity_improvement,
                'max_leaves': self.max_leaves,
                'max_depth': self.max_depth,
                'variable_dependencies': {var.name: [v.name for v in deps]
                                          for var, deps in self.variable_dependencies.items()},
                'leaves': [l.to_json() for l in self.leaves.values()],
                'innernodes': [n.to_json() for n in self.innernodes.values()],
                'priors': {varname: p.to_json() for varname, p in self.priors.items()},
                'root': ifnone(self.root, None, attrgetter('idx'))
                }

    @staticmethod
    def from_json(data: Dict[str, Any]):
        variables = OrderedDict([(d['name'], Variable.from_json(d)) for d in data['variables']])
        jpt = JPT(variables=list(variables.values()),
                  targets=[variables[v] for v in data['targets']] if data.get('targets') else None,
                  min_samples_leaf=data['min_samples_leaf'],
                  min_impurity_improvement=data['min_impurity_improvement'],
                  max_leaves=data['max_leaves'],
                  max_depth=data['max_depth'],
                  variable_dependencies={variables[var]: [variables[v] for v in deps]
                                         for var, deps in data['variable_dependencies'].items()}
                  )
        for d in data['innernodes']:
            DecisionNode.from_json(jpt, d)
        for d in data['leaves']:
            Leaf.from_json(jpt, d)
        jpt.priors = {varname: jpt.varnames[varname].domain.from_json(dist)
                      for varname, dist in data['priors'].items()}
        jpt.root = jpt.allnodes[data.get('root')] if data.get('root') is not None else None
        return jpt

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = JPT.from_json(state).__dict__

    def __eq__(self, o) -> bool:
        return (isinstance(o, JPT) and
                self.innernodes == o.innernodes and
                self.leaves == o.leaves and
                self.priors == o.priors and
                (self.dependency_matrix == o.dependency_matrix).all() and
                self.min_samples_leaf == o.min_samples_leaf and
                self.min_impurity_improvement == o.min_impurity_improvement and
                self.targets == o.targets and
                self.variables == o.variables and
                self.max_depth == o.max_depth and
                self.max_leaves == o.max_leaves)

    def encode(self, samples) -> np.array:
        """ Return a list of leaf indices that describe in what leaf the sample would land """
        result = np.zeros(len(samples))
        variable_index_map = VariableMap([(variable, idx) for (idx, variable) in enumerate(self.variables)])
        samples = self._preprocess_data(samples)
        for idx, leaf in self.leaves.items():
            contains = leaf.contains(samples, variable_index_map)
            result[contains == 1] = idx
        return result

    def infer(self, query, evidence=None, fail_on_unsatisfiability=True) -> Result:
        r'''For each candidate leaf ``l`` calculate the number of samples in which `query` is true:

        .. math::
            P(query|evidence) = \frac{p_q}{p_e}
            :label: query

        .. math::
            p_q = \frac{c}{N}
            :label: pq

        .. math::
            c = \frac{\prod{F}}{x^{n-1}}
            :label: c

        where ``Q`` is the set of variables in `query`, :math:`P_{l}` is the set of variables that occur in ``l``,
        :math:`F = \{v | v \in Q \wedge~v \notin P_{l}\}` is the set of variables in the `query` that do not occur in ``l``'s path,
        :math:`x = |S_{l}|` is the number of samples in ``l``, :math:`n = |F|` is the number of free variables and
        ``N`` is the number of samples represented by the entire tree.
        reference to :eq:`query`

        :param query:       the event to query for, i.e. the query part of the conditional P(query|evidence) or the prior P(query)
        :type query:        dict of {jpt.variables.Variable : jpt.learning.distributions.Distribution.value}
        :param evidence:    the event conditioned on, i.e. the evidence part of the conditional P(query|evidence)
        :type evidence:     dict of {jpt.variables.Variable : jpt.learning.distributions.Distribution.value}
        '''
        querymap = VariableMap()
        for key, value in query.items():
            querymap[key if isinstance(key, Variable) else self.varnames[key]] = value
        query_ = self._prepropress_query(querymap)
        evidencemap = VariableMap()
        if evidence:
            for key, value in evidence.items():
                evidencemap[key if isinstance(key, Variable) else self.varnames[key]] = value
        evidence_ = ifnone(evidencemap, {}, self._prepropress_query)

        r = Result(query_, evidence_)

        p_q = 0.
        p_e = 0.

        for leaf in self.apply(evidence_):
            p_m = 1
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                elif var.symbolic and var in leaf.path:
                    continue
                p_m *= leaf.distributions[var]._p(evidence_val)

            w = leaf.prior
            p_m *= w
            p_e += p_m

            if leaf.applies(query_):
                for var in set(query_.keys()):
                    query_val = query_[var]
                    if var.numeric and var in leaf.path:
                        query_val = query_val.intersection(leaf.path[var])
                    elif var.symbolic and var in leaf.path:
                        continue
                    p_m *= leaf.distributions[var]._p(query_val)
                p_q += p_m

                r.candidates.append(leaf)
                r.weights.append(p_m)

        if p_e == 0:
            if fail_on_unsatisfiability:
                raise ValueError('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence_))
            else:
                r.result = None
                r.weights = None
        else:
            r.result = p_q / p_e
            r.weights = [w / p_e for w in r.weights]
        return r

    def posterior(self, variables, evidence, fail_on_unsatisfiability=True) -> PosteriorResult:
        """
        :param variables:        the query variables of the posterior to be computed
        :type variables:         list of jpt.variables.Variable
        :param evidence:    the evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :type fail_on_unsatisfiability:  bool
        :return:            jpt.trees.InferenceResult containing distributions, candidates and weights
        """
        evidence_ = ifnone(evidence, {}, self._prepropress_query)
        result = PosteriorResult(variables, evidence_)
        variables = [self.varnames[v] if type(v) is str else v for v in variables]

        distributions = defaultdict(list)

        likelihoods = []
        priors = []

        for leaf in self.apply(evidence_):
            likelihood = 1
            # check if path of candidate leaf is consistent with evidence
            # (i.e. contains evicence variable with *correct* value or does not contain it at all)
            for var in set(evidence_.keys()):
                evidence_set = evidence_[var]
                if var in leaf.path:
                    evidence_set = evidence_set.intersection(leaf.path[var])
                likelihood *= leaf.distributions[var]._p(evidence_set)
            likelihoods.append(likelihood)
            priors.append(leaf.prior)

            for var in variables:
                evidence_set = evidence_.get(var)
                distribution = leaf.distributions[var]
                if evidence_set is not None:
                    if var in leaf.path:
                        evidence_set = evidence_set.intersection(leaf.path[var])
                        distribution = distribution.crop(evidence_set)
                distributions[var].append(distribution)

            result.candidates.append(leaf)

        weights = [l * p for l, p in zip(likelihoods, priors)]
        # result.result = sum(weights)
        try:
            weights = normalized(weights)
        except ValueError:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Evidence %s is unsatisfiable.' % format_path(evidence_))
            return None

        # initialize all query variables with None, in case dists
        # is empty (i.e. no candidate leaves -> query unsatisfiable)
        result.distributions = VariableMap()

        for var, dists in distributions.items():
            if var.numeric:
                result.distributions[var] = Numeric.merge(dists, weights=weights)
            elif var.symbolic:
                result.distributions[var] = Multinomial.merge(dists, weights=weights)

        return result

    def independent_marginals(self, variables: List[Variable], evidence: VariableMap, fail_on_unsatisfiability=True) ->\
            PosteriorResult or None:
        """ Compute the marginal distribution of every varialbe in 'variables' assuming independence.
        Unlike JPT.posterior, this method also can compute marginals on variables that are in the evidence.

        :param variables:        the query variables of the posterior to be computed
        :type variables:         list of jpt.variables.Variable
        :param evidence:    the evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :type fail_on_unsatisfiability:  bool
        :return:            jpt.trees.InferenceResult containing distributions, candidates and weights
        """

        # preprocess evidence
        evidence_ = ifnone(evidence, {}, self._prepropress_query)

        # construct result to save data in
        result = PosteriorResult(VariableMap.universe_map(variables), evidence_)

        # parse variables to variables if strings are given
        variables = [self.varnames[v] if type(v) is str else v for v in variables]

        # default map containing the distributions that need to be merged
        distributions = defaultdict(list)

        # list of weights that will determine how important the distributions are
        weights = []

        for leaf in self.leaves.values():
            leaf_prob = leaf.probability(evidence_)
            if leaf_prob <= 0.:
                continue
            weights.append(leaf_prob * leaf.prior)

            for var in variables:
                evidence_set = evidence_.get(var)
                distribution = leaf.distributions[var].copy()
                if evidence_set is not None:
                    if var in leaf.path:
                        evidence_set = evidence_set.intersection(leaf.path[var])
                        distribution = distribution.crop(evidence_set)
                distributions[var].append(distribution)

            result.candidates.append(leaf)

        weight_sum = sum(weights)
        result.result = weight_sum
        weights = [w/weight_sum for w in weights]

        if weight_sum == 0:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Evidence %s is unsatisfiable.' % format_path(evidence_))
            else:
                return None

        # initialize all query variables with None, in case dists
        # is empty (i.e. no candidate leaves -> query unsatisfiable)
        result.distributions = VariableMap()

        # for every variable and distribution that got selected
        for var, dists in distributions.items():
            # merge the distributions with weights according to their probability
            if var.numeric:
                result.distributions[var] = Numeric.merge(dists, weights=weights)
            elif var.symbolic:
                result.distributions[var] = Multinomial.merge(dists, weights=weights)

        return result

    def expectation(self, variables=None,
                    evidence=None,
                    confidence_level=None,
                    fail_on_unsatisfiability=True) -> ExpectationResult:
        '''
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.
        '''
        variables = ifnone([v if isinstance(v, Variable) else self.varnames[v] for v in variables],
                           set(self.variables) - set(evidence))
        posteriors = self.posterior(variables, evidence, fail_on_unsatisfiability=fail_on_unsatisfiability)
        conf_level = ifnone(confidence_level, .95)

        if posteriors is None:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence))
            else:
                return None

        final = VariableMap()
        for var, dist in posteriors.distributions.items():
            result = ExpectationResult(var, posteriors._evidence, conf_level)
            result._res = dist._expectation()
            result.candidates.extend(posteriors.candidates)
            if var.numeric:
                exp_quantile = dist.cdf.eval(result._res)
                result._lower = dist.ppf.eval(max(0., (exp_quantile - conf_level / 2.)))
                result._upper = dist.ppf.eval(min(1., (exp_quantile + conf_level / 2.)))
            final[var] = result
        return final

    def mpe(self, evidence=None, fail_on_unsatisfiability=True) -> MPEResult:
        '''
        Compute the (conditional) MPE state of the model.
        '''
        evidence_ = self._prepropress_query(evidence)
        distributions = {var: deque() for var in self.variables}

        r = MPEResult(evidence_)

        for leaf in self.apply(evidence_):
            p_m = 1
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                elif var.symbolic and var in leaf.path:
                    continue
                p_m *= leaf.distributions[var]._p(evidence_val)

            if not p_m: continue

            for var in self.variables:
                distributions[var].append((leaf.distributions[var], p_m))

        if not all([sum([w for _, w in distributions[var]]) for v in self.variables]):
            if fail_on_unsatisfiability:
                raise ValueError('Query is unsatisfiable: P(%s) is 0.' % var.str(evidence_val, fmt='logic'))
            else:
                return None

        posteriors = {var: var.domain.merge([d for d, _ in distributions[var]],
                                            normalized([w for _, w in distributions[var]]))
                      for var in distributions}

        for var, dist in posteriors.items():
            if var in evidence_:
                continue
            r.path.update({var: dist.mpe()})
        return r

    def _prepropress_query(self, query, transform_values=True, remove_none=True) -> VariableMap:
        '''
        Transform a query entered by a user into an internal representation
        that can be further processed.
        '''
        # Transform lists into a numeric interval:
        query_ = VariableMap()
        # Transform single numeric values in to intervals given by the haze
        # parameter of the respective variable:
        for key, arg in query.items():
            if arg is None and remove_none:
                continue
            var = key if isinstance(key, Variable) else self.varnames[key]
            if var.numeric:
                if type(arg) is list:
                    arg = list2interval(arg)
                if isinstance(arg, numbers.Number) and transform_values:
                    val = var.domain.values[arg]
                    prior = self.priors[var.name]
                    quantile = prior.cdf.functions[max(1, min(len(prior.cdf) - 2,
                                                              prior.cdf.idx_at(val)))].eval(val)
                    lower = quantile - var.haze / 2
                    upper = quantile + var.haze / 2
                    query_[var] = ContinuousSet(prior.ppf.functions[max(1,
                                                                        min(len(prior.cdf) - 2,
                                                                            prior.ppf.idx_at(lower)))].eval(lower),
                                                prior.ppf.functions[min(len(prior.ppf) - 2,
                                                                        max(1,
                                                                            prior.ppf.idx_at(upper)))].eval(upper))
                elif isinstance(arg, ContinuousSet) and transform_values:
                    query_[var] = ContinuousSet(var.domain.values[arg.lower],
                                                var.domain.values[arg.upper], arg.left, arg.right)
                else:
                    query_[var] = arg
            if var.symbolic:
                # Transform into internal values (symbolic values to their indices):
                if type(arg) is not set:
                    arg = {arg}
                query_[var] = {var.domain.values[v] if transform_values else v for v in arg}

        JPT.logger.debug('Original :', pprint.pformat(query), '\nProcessed:', pprint.pformat(query_))
        return query_

    def apply(self, query):
        # if the sample doesn't match the features of the tree, there is no valid prediction possible
        if not set(query.keys()).issubset(set(self._variables)):
            raise TypeError(f'Invalid query. Query contains variables that are not '
                            f'represented by this tree: {[v for v in query.keys() if v not in self._variables]}')

        # find the leaf (or the leaves) that have each variable either
        # - not occur in the path to this node OR
        # - match the boolean/symbolic value in the path OR
        # - lie in the interval of the numeric value in the path
        # -> return leaf that matches query
        yield from (leaf for leaf in self.leaves.values() if leaf.applies(query))

    def leaf_distribution(self) -> jpt.variables.SymbolicVariable:
        """Generate a multinomial distribution with the leaves as variables and their weights as probabilities. """
        leaf_variable = jpt.variables.SymbolicVariable("Leaf", SymbolicType("Leaf", self.leaves.keys()))
        leaf_variable = leaf_variable.dist([leaf.prior for leaf in self.leaves.values()])
        return leaf_variable

    def multiply_by_leaf_prior(self, leaf_prior: Dict[int, float]):
        """Include a different prior for leaves by multiplying and normalizing both priors."""
        result = self.copy()
        for idx, leaf in result.leaves.items():
            result.leaves[idx].prior *= leaf_prior[idx]

        probability_mass = sum(leaf.prior for leaf in result.leaves.values())

        for idx, leaf in result.leaves.items():
            result.leaves[idx].prior /= probability_mass

        return result

    def c45(self, data, start, end, parent, child_idx, depth) -> None:
        '''
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.

        :param data:        the indices for the training samples used to calculate the gain.
        :param start:       the starting index in the data.
        :param end:         the stopping index in the data.
        :param parent:      the parent node of the current iteration, initially ``None``.
        :param child_idx:   the index of the child in the current iteration.
        :param depth:       the depth of the tree in the current recursion level.
        '''
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = ifnone(self.min_impurity_improvement, 0)
        n_samples = end - start
        split_var_idx = split_pos = -1
        split_var = None
        impurity = self.impurity

        max_gain = impurity.compute_best_split(start, end)
        JPT.logger.debug('Data range: %d-%d,' % (start, end),
                         'split var:', split_var,
                         ', split_pos:', split_pos,
                         ', gain:', max_gain)

        if max_gain:
            split_pos = impurity.best_split_pos
            split_var_idx = impurity.best_var
            split_var = self.variables[split_var_idx]

        if max_gain <= min_impurity_improvement or depth >= self.max_depth:  # -----------------------------------------
            leaf = node = Leaf(idx=len(self.allnodes), parent=parent)

            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.dist(data=data,
                                               rows=self.indices[start:end], col=i)

            leaf.prior = n_samples / data.shape[0]
            leaf.samples = n_samples

            self.leaves[leaf.idx] = leaf

        else:  # -------------------------------------------------------------------------------------------------------
            node = DecisionNode(idx=len(self.allnodes),
                                variable=split_var,
                                parent=parent)
            node.samples = n_samples
            self.innernodes[node.idx] = node

            if split_var.symbolic:  # ----------------------------------------------------------------------------------
                split_value = int(data[self.indices[start + split_pos], split_var_idx])
                splits = [{split_value},
                          set(split_var.domain.values.values()) - {split_value}]

            elif split_var.numeric:  # ---------------------------------------------------------------------------------
                split_value = (data[self.indices[start + split_pos], split_var_idx] +
                               data[self.indices[start + split_pos + 1], split_var_idx]) / 2
                splits = [Interval(np.NINF, split_value, EXC, EXC),
                          Interval(split_value, np.PINF, INC, EXC)]

            else:  # ---------------------------------------------------------------------------------------------------
                raise TypeError('Unknown variable type: %s.' % type(split_var).__name__)

            self.c45queue.append((data, start, start + split_pos + 1, node, 0, depth + 1))
            self.c45queue.append((data, start + split_pos + 1, end, node, 1, depth + 1))

            node.splits = splits

        JPT.logger.debug('Created', str(node))

        if parent is not None:
            parent.set_child(child_idx, node)

        if self.root is None:
            self.root = node

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}\n'
                f'{self.pfmt()}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}\n'
                f'{self.pfmt()}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def pfmt(self) -> str:
        '''Return a pretty-format string representation of this JPT.'''
        return self._pfmt(self.root, 0)

    def _pfmt(self, node, indent) -> str:
        return "{}{}\n{}".format(" " * indent,
                                 str(node),
                                 ''.join([self._pfmt(c, indent + 4) for c in node.children])
                                 if isinstance(node, DecisionNode) else '')

    def _preprocess_data(self, data=None, rows=None, columns=None) -> np.ndarray:
        '''
        Transform the input data into an internal representation.
        '''
        if sum(d is not None for d in (data, rows, columns)) > 1:
            raise ValueError('Only either of the three is allowed.')
        elif sum(d is not None for d in (data, rows, columns)) < 1:
            raise ValueError('No data pased.')

        JPT.logger.info('Preprocessing data...')

        if isinstance(data, np.ndarray) and data.shape[0] or isinstance(data, list):
            rows = data

        if isinstance(rows, list) and rows:  # Transpose the rows
            columns = [[row[i] for row in rows] for i in range(len(self.variables))]
        elif isinstance(rows, np.ndarray) and rows.shape[0]:
            columns = rows.T

        if isinstance(columns, list) and columns:
            shape = len(columns[0]), len(columns)
        elif isinstance(columns, np.ndarray) and columns.shape:
            shape = columns.T.shape
        elif isinstance(data, pd.DataFrame):
            shape = data.shape
        else:
            raise ValueError('No data given.')

        data_ = np.ndarray(shape=shape, dtype=np.float64, order='C')
        if isinstance(data, pd.DataFrame):
            if set(self.varnames).symmetric_difference(set(data.columns)):
                raise ValueError('Unknown variable names: %s'
                                 % ', '.join(mapstr(set(self.varnames).symmetric_difference(set(data.columns)))))

            # Check if the order of columns in the data frame is the same
            # as the order of the variables.
            if not all(c == v for c, v in zip_longest(data.columns, self.varnames)):
                raise ValueError('Columns in DataFrame must coincide with variable order: %s' %
                                 ', '.join(mapstr(self.varnames)))
            transformations = {v: self.varnames[v].domain.values.transformer() for v in data.columns}
            try:
                data_[:] = data.transform(transformations).values
            except ValueError:
                err(transformations)
                raise
        else:
            for i, (var, col) in enumerate(zip(self.variables, columns)):
                data_[:, i] = [var.domain.values[v] for v in col]
        return data_

    def learn(self, data=None, rows=None, columns=None) -> 'JPT':
        '''Fits the ``data`` into a regression tree.

        :param data:    The training examples (assumed in row-shape)
        :type data:     [[str or float or bool]]; (according to `self.variables`)
        :param rows:    The training examples (assumed in row-shape)
        :type rows:     [[str or float or bool]]; (according to `self.variables`)
        :param columns: The training examples (assumed in row-shape)
        :type columns:  [[str or float or bool]]; (according to `self.variables`)
        '''
        with _lock:
            # ----------------------------------------------------------------------------------------------------------
            # Check and prepare the data
            global _data
            _data = self._preprocess_data(data=data, rows=rows, columns=columns)
            if _data.shape[0] < 1:
                raise ValueError('No data for learning.')

            self.indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
            self.indices[0] = 0
            np.cumsum(self.indices, out=self.indices)
            # Initialize the impurity calculation
            self.impurity = Impurity(self)
            self.impurity.setup(_data, self.indices)
            self.impurity.min_samples_leaf = max(1, self.min_samples_leaf)

            JPT.logger.info('Data transformation... %d x %d' % _data.shape)

            # ----------------------------------------------------------------------------------------------------------
            # Initialize the internal data structures
            self._reset()

            # ----------------------------------------------------------------------------------------------------------
            # Determine the prior distributions
            started = datetime.datetime.now()
            JPT.logger.info('Learning prior distributions...')
            self.priors = {}
            pool = mp.Pool()
            for i, prior in enumerate(pool.map(_prior, [(i, var.to_json()) for i, var in enumerate(
                    self.variables)])):  # {var: var.dist(data=data[:, i]) }
                self.priors[self.variables[i].name] = self.variables[i].domain.from_json(prior)
            JPT.logger.info('Prior distributions learnt in %s.' % (datetime.datetime.now() - started))
            # self.impurity.priors = [self.priors[v.name] for v in self.variables if v.numeric]
            pool.close()
            pool.join()

            # ----------------------------------------------------------------------------------------------------------
            # Start the training

            started = datetime.datetime.now()
            JPT.logger.info('Started learning of %s x %s at %s '
                            'requiring at least %s samples per leaf' % (_data.shape[0],
                                                                        _data.shape[1],
                                                                        started,
                                                                        int(self.impurity.min_samples_leaf)))
            learning = GENERATIVE if self.targets is None else DISCRIMINATIVE
            JPT.logger.info('Learning is %s. ' % learning)
            if learning == DISCRIMINATIVE:
                JPT.logger.info('Target variables (%d): %s\n'
                                'Feature variables (%d): %s' % (len(self.targets),
                                                                ', '.join(mapstr(self.targets)),
                                                                len(self.variables) - len(self.targets),
                                                                ', '.join(
                                                                    mapstr(set(self.variables) - set(self.targets)))))
            # build up tree
            self.c45queue.append((_data, 0, _data.shape[0], None, None, 0))
            while self.c45queue:
                self.c45(*self.c45queue.popleft())

            # ----------------------------------------------------------------------------------------------------------
            # Print the statistics

            JPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
            # if logger.level >= 20:
            JPT.logger.debug(self)
            return self

    fit = learn

    @property
    def min_samples_leaf(self):
        if type(self._min_samples_leaf) is int:
            return self._min_samples_leaf
        if type(self._min_samples_leaf) is float and 0 < self._min_samples_leaf < 1:
            if _data is None:
                return self._min_samples_leaf
            else:
                return int(self._min_samples_leaf * len(_data))
        return int(self._min_samples_leaf)

    @staticmethod
    def sample(sample, ft):
        # NOTE: This sampling is NOT uniform for intervals that are infinity in any direction! TODO: FIX to sample from CATEGORICAL
        if ft not in sample:
            return Interval(np.NINF, np.inf, EXC, EXC).sample()
        else:
            iv = sample[ft]

        if isinstance(iv, Interval):
            if iv.lower == -np.inf and iv.upper == np.inf:
                return Interval(np.NINF, np.inf, EXC, EXC).sample()
            if iv.lower == -np.inf:
                if any([i.right == EXC for i in iv.intervals]):
                    # workaround to be able to sample from open interval
                    return iv.upper - 0.01 * iv.upper
                else:
                    return iv.upper
            if iv.upper == np.inf:
                # workaround to be able to sample from open interval
                if any([i.left == EXC for i in iv.intervals]):
                    return iv.lower + 0.01 * iv.lower
                else:
                    return iv.lower

            return iv.sample()
        else:
            return iv

    def likelihood(self, queries: np.ndarray, dirac_scaling=2., min_distances=None) -> np.ndarray:
        """Get the probabilities of a list of worlds. The worlds must be fully assigned with
        single numbers (no intervals).

        :param queries: An array containing the worlds. The shape is (x, len(variables)).
        :type queries: np.array
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :type dirac_scaling: float
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :type min_distances: Dict[Variable, float]
        Returns: An np.array with shape (x, ) containing the probabilities.
        """
        # create minimal distances for each numeric variable such a senseful metric can be computed if not provided
        if min_distances is None:
            min_distances: Dict[Variable, float] = dict()
            for idx, variable in enumerate(self.variables):
                if variable.numeric:
                    samples = np.unique(queries[:, idx])
                    distances = np.diff(samples)
                    min_distances[variable] = min(distances) if len(distances) > 0 else dirac_scaling

        for idx, variable in enumerate(self.variables):
            # convert the symbolic columns to the representation used in jpts
            if variable.symbolic:
                for value, label in zip(variable.domain.values, variable.domain.labels):
                    queries[queries[:, idx] == value, idx] = label

            # scale numeric variables if needed
            elif variable.numeric and issubclass(variable.domain, ScaledNumeric):
                queries[:, idx] = variable.domain.scaler.transform(queries[:, idx])

        # initialize probabilities
        probabilities = np.zeros(len(queries))

        # for all leaves
        for leaf in self.leaves.values():
            leaf_probabilities = leaf.parallel_likelihood(queries, dirac_scaling, min_distances)
            probabilities = probabilities + leaf_probabilities
        return probabilities

    def reverse(self, query, confidence=.5) -> List[Tuple[Dict, List[Node]]]:
        '''Determines the leaf nodes that match query best and returns their respective paths to the root node.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
        :type query: dict
        :param confidence:  the confidence level for this MPE inference
        :type confidence: float
        :returns: a mapping from probabilities to lists of matcalo.core.algorithms.JPT.Node (path to root)
        :rtype: dict
        '''
        # if none of the target variables is present in the query, there is no match possible
        if set(query.keys()).isdisjoint(set(self.variables)):
            return []

        # Transform into internal values/intervals (symbolic values to their indices) and update to contain all possible variables
        query = {var: list2interval(val) if type(val) in (list, tuple) and var.numeric else val if type(val) in (
        list, tuple) else [val] for var, val in query.items()}
        query_ = {var: set(var.domain.value[v] for v in val) for var, val in query.items()}
        for i, var in enumerate(self.variables):
            if var in query_: continue
            if var.numeric:
                query_[var] = list2interval([np.NINF, np.PINF])
            else:
                query_[var] = var.domain.values

        # find the leaf (or the leaves) that matches the query best
        confs = {}
        for k, l in self.leaves.items():
            confs_ = defaultdict(float)
            for v, dist in l.distributions.items():
                if v.numeric:
                    confs_[v] = dist.p(query_[v])
                else:
                    conf = 0.
                    for sv in query_[v]:
                        conf += dist.p(sv)
                    confs_[v] = conf
            confs[l] = confs_

        # the candidates are the one leaves that satisfy the confidence requirement (i.e. each free variable of a leaf must satisfy the requirement)
        candidates = sorted([leaf for leaf, confs in confs.items() if all(c >= confidence for c in confs.values())],
                            key=lambda l: sum(confs[l].values()), reverse=True)

        # for the chosen candidate determine the path to the root
        paths = []
        for c in candidates:
            p = []
            curcand = c
            while curcand is not None:
                p.append(curcand)
                curcand = curcand.parent
            paths.append((confs[c], p))

        # elements of path are tuples (a, b) with a being mappings of {var: confidence} and b being an ordered list of
        # nodes representing a path from a leaf to the root
        return paths

    def plot(self, title=None, filename=None, directory='/tmp', plotvars=None, view=True, max_symb_values=10):
        '''Generates an SVG representation of the generated regression tree.

        :param title:   (str) title of the plot
        :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
        :type filename: str
        :param directory: the location to save the SVG file to
        :type directory: str
        :param plotvars: the variables to be plotted in the graph
        :type plotvars: <jpt.variables.Variable>
        :param view: whether the generated SVG file will be opened automatically
        :type view: bool
        :param max_symb_values: limit the maximum number of symbolic values to this number
        '''
        if plotvars is None:
            plotvars = []
        plotvars = [self.varnames[v] if type(v) is str else v for v in plotvars]

        title = ifnone(title, 'unnamed')

        if not os.path.exists(directory):
            os.makedirs(directory)

        dot = Digraph(format='svg', name=filename or title,
                      directory=directory,
                      filename=f'{filename or title}.dot')

        # create nodes
        sep = ",<BR/>"
        for idx, n in self.allnodes.items():
            imgs = ''

            # plot and save distributions for later use in tree plot
            if isinstance(n, Leaf):
                rc = math.ceil(math.sqrt(len(plotvars)))
                img = ''
                for i, pvar in enumerate(plotvars):
                    img_name = html.escape(f'{pvar.name}-{n.idx}')

                    params = {} if pvar.numeric else {'horizontal': True,
                                                      'max_values': max_symb_values}

                    n.distributions[pvar].plot(title=html.escape(pvar.name),
                                               fname=img_name,
                                               directory=directory,
                                               view=False,
                                               **params)
                    img += (f'''{"<TR>" if i % rc == 0 else ""}
                                        <TD><IMG SCALE="TRUE" SRC="{os.path.join(directory, f"{img_name}.png")}"/></TD>
                                {"</TR>" if i % rc == rc - 1 or i == len(plotvars) - 1 else ""}
                                ''')

                    # clear current figure to allow for other plots
                    plt.clf()

                if plotvars:
                    imgs = f'''
                                <TR>
                                    <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2">
                                        <TABLE>
                                            {img}
                                        </TABLE>
                                    </TD>
                                </TR>
                                '''

            land = '<BR/>\u2227 '
            element = ' \u2208 '

            # content for node labels
            nodelabel = f'''<TR>
                                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{"Leaf" if isinstance(n, Leaf) else "Node"} #{n.idx}</B><BR/>{html.escape(n.str_node)}</TD>
                            </TR>'''

            if isinstance(n, Leaf):
                nodelabel = f'''{nodelabel}{imgs}
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{n.samples} ({n.prior * 100:.3f}%)</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{html.escape(v.name)}=' + (f'{html.escape(str(dist.expectation()))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items() if self.targets is None or v in self.targets])}</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE"><B>path:</B></TD>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE">{f"{land}".join([html.escape(var.str(val, fmt='set')) for var, val in n.path.items()])}</TD>
                                </TR>
                                '''

            # stitch together
            lbl = f'''<<TABLE ALIGN="CENTER" VALIGN="MIDDLE" BORDER="0" CELLBORDER="0" CELLSPACING="0">
                            {nodelabel}
                      </TABLE>>'''

            if isinstance(n, Leaf):
                dot.node(str(idx),
                         label=lbl,
                         shape='box',
                         style='rounded,filled',
                         fillcolor=green)
            else:
                dot.node(str(idx),
                         label=lbl,
                         shape='ellipse',
                         style='rounded,filled',
                         fillcolor=orange)

        # create edges
        for idx, n in self.innernodes.items():
            for i, c in enumerate(n.children):
                if c is None: continue
                dot.edge(str(n.idx), str(c.idx), label=html.escape(n.str_edge(i)))

        # show graph
        JPT.logger.info(f'Saving rendered image to {os.path.join(directory, filename or title)}.svg')

        # improve aspect ratio of graph having many leaves or disconnected nodes
        dot = dot.unflatten(stagger=3)
        dot.render(view=view, cleanup=False)

    def pickle(self, fpath) -> None:
        '''Pickles the fitted regression tree to a file at the given location ``fpath``.

        :param fpath: the location for the pickled file
        :type fpath: str
        '''
        with open(os.path.abspath(fpath), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fpath):
        '''Loads the pickled regression tree from the file at the given location ``fpath``.

        :param fpath: the location of the pickled file
        :type fpath: str
        '''
        with open(os.path.abspath(fpath), 'rb') as f:
            try:
                JPT.logger.info(f'Loading JPT {os.path.abspath(fpath)}')
                return pickle.load(f)
            except ModuleNotFoundError:
                JPT.logger.error(f'Could not load file {os.path.abspath(fpath)}')
                raise Exception(f'Could not load file {os.path.abspath(fpath)}. Probably deprecated.')

    @staticmethod
    def calcnorm(sigma, mu, intervals):
        '''Computes the CDF for a multivariate normal distribution.

        :param sigma: the standard deviation
        :param mu: the expected value
        :param intervals: the boundaries of the integral
        :type sigma: float
        :type mu: float
        :type intervals: list of matcalo.utils.utils.Interval
        '''
        from scipy.stats import mvn
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], mu, sigma))

    def copy(self):
        """Return a new copy of this jpt where all references are the original tree are cut."""
        return JPT.from_json(self.to_json())

    def conditional_jpt(self, evidence: VariableMap):
        """
        Apply evidence on a JPT and get a new JPT that represent P(x|evidence).
        The new JPT contains all variables that are not in the evidence and is a 
        full joint probability distribution over those variables.

        :param evidence: A variable Map mapping the observed variables to there observed,
            single values (not intervals)
        :type evidence: ``VariableMap``
        """

        # the new jpt that acts as conditional joint probability distribution
        conditional_jpt: JPT = self.copy()

        if len(evidence) == 0:
            return conditional_jpt

        evidence = self._prepropress_query(evidence, transform_values=False)
        unvisited_nodes = queue.Queue()
        unvisited_nodes.put_nowait(conditional_jpt.allnodes[self.root.idx])

        while not unvisited_nodes.empty():

            # get the next node to inspect
            current_node: Node = unvisited_nodes.get_nowait()

            # if it is a leaf skip this iteration
            if isinstance(current_node, Leaf):
                current_node: Leaf
                probability = current_node.probability(evidence)
                current_node.prior = probability
                continue

            # syntax highlighting
            current_node: DecisionNode

            # remember the indices of the nodes that need to get removed
            invalid = []

            # check if the children of the node need to be traversed
            for idx, child in enumerate(current_node.children):

                # traverse consistent children
                if child.consistent_with(evidence):
                    unvisited_nodes.put_nowait(child)

                # mark invalid children for removal
                else:
                    invalid = [idx] + invalid

            # remove invalid children from the tree and the children list
            for idx in invalid:
                # get all the indices of the subtree members
                removable_indices = [node.idx for node in
                                     current_node.children[idx].recursive_children()] + \
                                    [current_node.children[idx].idx]

                # for all dead nodes 
                for jdx in removable_indices:
                    # if it is a leaf remove it from the leaves
                    if isinstance(self.allnodes[jdx], Leaf):
                        del conditional_jpt.leaves[jdx]
                    # if it is an inner node remove it from the inner nodes
                    else:
                        del conditional_jpt.innernodes[jdx]

                # remove it as child
                del current_node.children[idx]

        # calculate remaining probability mass
        probability_mass = sum(leaf.prior for leaf in conditional_jpt.leaves.values())

        # clean up not needed distributions and redistribute probability mass
        for leaf in conditional_jpt.leaves.values():
            leaf.prior /= probability_mass
            for variable, value in evidence.items():
                # adjust leaf distributions
                leaf.distributions[variable] = leaf.distributions[variable].apply_restriction(value)

        # clean up not needed path restrictions
        for node in conditional_jpt.allnodes.values():
            for variable in evidence.keys():
                if variable in node.path.keys():
                    del node.path[variable]

        return conditional_jpt

    def conditional_jpt_safe(self, evidence: VariableMap):
        result = self.copy()

        if len(evidence) == 0:
            return result

        for idx, leaf in result.leaves.items():
            print(evidence)
            print(leaf.probability(evidence))
            result.leaves[idx].prior *= leaf.probability(evidence)

        result.plot()
        exit()
        result = result.normalize()
        return result
    def normalize(self):
        probability_mass = sum(leaf.prior for leaf in self.leaves.values())
        for idx, leaf in self.leaves.items():
            self.leaves[idx].prior /= probability_mass
        return self

    def save(self, file) -> None:
        '''
        Write this JPT persistently to disk.

        ``file`` can be either a string or file-like object.
        '''
        if type(file) is str:
            with open(file, 'w+') as f:
                json.dump(self.to_json(), f)
        else:
            json.dump(self.to_json(), file)

    @staticmethod
    def load(file):
        '''
        Load a JPT from disk.
        '''
        if type(file) is str:
            with open(file, 'r') as f:
                t = json.load(f)
        else:
            t = json.load(file)
        return JPT.from_json(t)

    def depth(self):
        """Calculate the maximal depth of a leaf in the current tree."""
        return max([leaf.depth() for leaf in self.leaves.values()])

    def total_samples(self):
        """Calculate the total number of samples represented by this tree."""
        return sum(l.samples for l in self.leaves.values())

    def marginal_jpt(self, marginal_variables: List[jpt.variables.Variable]):
        """ Create a marginal joint probability distribution over all 'marginal_variables'.
        This is done by inducing a new tree that reduces variance on the marginals by calculating the variances giving
        the original distribution (jpt).
        All possible splits are given by all splits in this tree since they are the only meaningful dependencies.
        @param marginal_variables:
        @return: a new JPT
        """

        # collect all splits
        all_splits = self.all_splits(marginal_variables)

        # calculate variable dependencies on marginal tree
        remaining_dependencies = VariableMap()
        for variable in marginal_variables:
            remaining_dependencies[variable] = [v for v in self.variable_dependencies[variable]
                                                if v in marginal_variables and v.name != variable.name]

        # construct empty marginal tree
        marginal_jpt = JPT(marginal_variables, min_samples_leaf=self.min_samples_leaf,
                           min_impurity_improvement=self.min_impurity_improvement, max_leaves=self.max_leaves,
                           max_depth=self.max_depth, variable_dependencies=remaining_dependencies)

        # initialize root node for the marginal
        root_node = DecisionNode(1, None)

        # initialize impurities to know when to stop
        impurities: Dict[int, float] = dict()
        impurities[1] = self.posterior(marginal_variables, VariableMap()).impurity()

        # initialize queue with root
        unexpanded_nodes = deque()
        unexpanded_nodes.append(root_node)

        # while there is still stuff to expand
        while len(unexpanded_nodes) > 0:

            # get the current node
            current_node = unexpanded_nodes.pop()

            expansions_result = self.expand_node(current_node, JPT.possible_splits_of_node(current_node, all_splits),
                                                 marginal_jpt.variable_dependencies, impurities[current_node.idx])

            # if the expansion does not yield a usable result
            if expansions_result is None:
                # add node to leaves
                marginal_jpt.leaves[current_node.idx] = current_node

                # terminate the recursion here
                continue

            # otherwise unpack the result
            else:
                left_node, right_node, left_impurity, right_impurity = expansions_result

                # save impurities
                impurities[left_node.idx] = left_impurity
                impurities[right_node.idx] = right_impurity

                # set children and nodes
                current_node.children = [left_node, right_node]
                marginal_jpt.innernodes[current_node.idx] = current_node

            # check if the left node can be further expanded
            if left_node.samples >= 2 * marginal_jpt.min_samples_leaf \
                    and impurities[left_node.idx] > marginal_jpt.min_impurity_improvement:
                # add node to the ones that need to be expanded
                unexpanded_nodes.append(left_node)

            # if it cannot lead to a meaningful split
            else:
                # add left node to leaves
                marginal_jpt.leaves[left_node.idx] = left_node

            # check if the right node can be further expanded
            if right_node.samples >= 2 * marginal_jpt.min_samples_leaf \
                    and impurities[right_node.idx] > marginal_jpt.min_impurity_improvement:
                unexpanded_nodes.append(right_node)

            # if it cannot lead to a meaningful split
            else:
                # add right node to leaves
                marginal_jpt.leaves[right_node.idx] = right_node

        self.create_leaves(marginal_jpt)
        return marginal_jpt

    def expand_node(self, node: Node, all_splits: VariableMap, variable_dependencies: VariableMap,
                    node_impurity: float):
        """
        Expand a node and return the best subpartitions of that node or none if there are no valid partitions.
        @param node: The node to split
        @param all_splits: A VariableMap describing each possible split for each variable
        @param variable_dependencies: A VariableMap similar to self.variable_dependencies
        @param node_impurity: The impurity of the node given in
        @return: None if there are no good splits or left_child, right_child, left_impurity, right_impurity
        """
        # create the path as variable map
        path = node.path

        marginal_variables: List[Variable] = list(all_splits.keys())

        # array to save the best split for each node in
        best_splits = np.full((len(all_splits), 5), float("inf"))

        for variable_idx, (variable, splits) in enumerate(all_splits.items()):

            # skip variables that don't influence any other variable
            if len(variable_dependencies[variable]) == 0 or len(splits) == 0:
                continue

            # list to store impurities
            impurities = np.full((len(splits), 2), float("inf"))

            # list to store number of samples in those splits
            lengths = np.zeros((len(splits), 2))

            # for every split
            for split_idx, split in enumerate(splits):
                if variable.numeric:
                    # construct positive and negative variable
                    positive_path = path.copy()
                    negative_path = path.copy()
                    if variable not in path:
                        positive_path[variable] = ContinuousSet(-float("inf"), split)
                        negative_path[variable] = ContinuousSet(split, float("inf"))
                    else:
                        positive_path[variable] = positive_path[variable].intersection(
                            ContinuousSet(-float("inf"), split))
                        negative_path[variable] = negative_path[variable].intersection(
                            ContinuousSet(split, float("inf")))

                elif variable.symbolic:
                    # TODO create sets lol
                    pass

                # get independent marginals of remaining data and its length
                positive_result = self.independent_marginals(marginal_variables, positive_path, False)
                negative_result = self.independent_marginals(marginal_variables, negative_path, False)

                if positive_result is None or negative_result is None:
                    # store impurity and number of samples
                    impurities[split_idx] = (float("inf"),
                                             float("inf"))
                    lengths[split_idx] = (0,
                                          0)
                else:
                    # store impurity and number of samples
                    impurities[split_idx] = (positive_result.impurity(variable_dependencies[variable]),
                                             negative_result.impurity(variable_dependencies[variable]))
                    lengths[split_idx] = (positive_result.result * self.total_samples(),
                                          negative_result.result * self.total_samples())

            # get splits that would result in too few samples and set the corresponding impurities to infinity
            invalid_splits, _ = np.where(lengths < self.min_samples_leaf)
            impurities[invalid_splits] = (float("inf"), float("inf"))

            # get the best split index
            best_split_idx = np.argmin(impurities)
            best_split_idx = np.unravel_index(best_split_idx, impurities.shape)

            # save the best split and its results on this variable to the array
            best_splits[variable_idx] = (splits[best_split_idx[0]], impurities[best_split_idx[0], 0],
                                         impurities[best_split_idx[0], 1],
                                         lengths[best_split_idx[0], 0],
                                         lengths[best_split_idx[0], 1])

        # get best split parameters among all variables
        best_split = np.argmin(best_splits[:, 1:3])
        best_split_idx = np.array(np.unravel_index(best_split, best_splits[:, 1:3].shape))

        # counter the shift that is obtained by unraveling in a view
        best_split_idx[1] += 1
        best_split_idx = tuple(best_split_idx)

        # return None if this solution is not feasible
        if best_splits[best_split_idx] == float("inf") \
                or node_impurity - best_splits[best_split_idx] <= self.min_impurity_improvement:
            return None

        # construct left (positive) node
        left_node = jpt.trees.DecisionNode(2 * node.idx, marginal_variables[best_split_idx[0]], node)
        left_node.samples = best_splits[best_split_idx[0], 3]
        left_node._path = node._path.copy()

        # construct right (negative) node
        right_node = jpt.trees.DecisionNode(2 * node.idx + 1, marginal_variables[best_split_idx[0]], node)
        right_node.samples = best_splits[best_split_idx[0], 4]
        right_node._path = node._path.copy()

        # create the decision criterion and apply it ot the path
        split_value = best_splits[best_split_idx[0], 0]

        # create the splits for the parent node
        if marginal_variables[best_split_idx[0]].numeric:
            splits = [Interval(np.NINF, split_value, EXC, EXC),
                      Interval(split_value, np.PINF, INC, EXC)]

        elif marginal_variables[best_split_idx[0]].symbolic:
            splits = [{int(split_value)},
                      set(marginal_variables[best_split_idx[0]].domain.values.values()) - {int(split_value)}]

        # update splits and splitting variable in parent node
        node.variable = marginal_variables[best_split_idx[0]]
        node.splits = splits

        # extend path of both nodes with restrictions
        left_node._path.append((marginal_variables[best_split_idx[0]], splits[0]))
        right_node._path.append((marginal_variables[best_split_idx[0]], splits[1]))

        # return left child, right child, left impurity, right impurity
        return left_node, right_node, best_splits[best_split_idx[0], 1], best_splits[best_split_idx[0], 2]

    def create_leaves(self, marginal_tree):
        """Create the probability distributions and leaf nodes for the marginal tree.

        :param marginal_tree: The marginal tree where the leafs should be calculated from this tree
        :type marginal_tree: JPT
        """

        # for every leaf node (that perhaps is a DecisionNode)
        for leaf_idx, leaf in marginal_tree.leaves.items():

            # calculate the distributions from this tree
            distributions = self.independent_marginals(marginal_tree.variables, leaf.path)

            # create new leaf
            new_leaf = Leaf(leaf.idx, leaf.parent, self.total_samples() * distributions.result)

            # set distributions
            new_leaf.distributions = distributions.distributions

            # set leaf
            marginal_tree.leaves[leaf_idx] = new_leaf

        return marginal_tree

    @staticmethod
    def possible_splits_of_node(node: Node, all_splits: VariableMap) -> VariableMap:
        """
        Calculate the splits that can be done from  a node given a set of possible splits.
        @param node: The node in question
        @param all_splits: the splits that can be made overall
        @return: the splits that can be made at this node
        """

        # create resulting map
        result = VariableMap()

        # load path of the given node
        path = node.path

        # for every variable and its possible splits in all splits
        for variable, splits in all_splits.items():

            # if variable is not limited by the nodes path
            if variable not in path.keys():

                # set all splits as possible
                result[variable] = splits

            elif variable.numeric:

                # get numeric splits in range of this node
                result[variable] = [split for split in splits if path[variable].lower < split < path[variable].upper]

            elif variable.symbolic:

                # get numeric splits in range of this node
                result[variable] = [split for split in splits if split in path[variable]]

        return result

    def all_splits(self, variables: List[Variable], as_lists: bool = True) -> VariableMap:
        """ Generate a variable map containing all splits that are made in the tree.
            The splits are sorted (if lists) and unique for each variable.

            @param variables: The variables to collect the splits for
            @param as_lists: Rather to return a VariableMap of lists or sets
            """

        # construct a map for every remaining variable that contains all splits
        all_splits = VariableMap([(v, set()) for v in variables])

        # fill splits
        for leaf in self.leaves.values():
            for variable in variables:

                # skip if there is no influence of the current variable in this leaf
                if variable not in leaf.path.keys():
                    continue

                # get the restriction of the current leaf on that variable
                restriction = leaf.path[variable]

                # if its numeric
                if variable.numeric:

                    # add the finite values to the set of possible splits
                    all_splits[variable].update(value for value in [restriction.lower, restriction.upper]
                                                if value != float("inf") and value != -float("inf"))

                # add all symbolic dependencies to the possibilities
                elif variable.symbolic:
                    all_splits[variable].update(restriction)

        # convert splits from sets to lists if desired
        if as_lists:
            for variable, splits in all_splits.items():
                all_splits[variable] = list(splits)

        return all_splits

    def postprocess_leaves(self):
        """ Postprocess the tree such that every point in the convex hull has
            a probability greater than 0. This only changes the numeric distributions. """

        # get total number of samples and use 1/total as default value
        total_samples = self.total_samples()

        # for every leaf
        for idx, leaf in self.leaves.items():
            # for numeric every distribution
            for variable, distribution in leaf.distributions.items():
                if variable.numeric and variable in leaf.path.keys() and not distribution.is_dirac_impulse():
                    print(type(distribution))
                    # if the leaf is not the "lowest" in this dimension
                    if leaf.path[variable].lower > -float("inf"):
                        # create uniform distribution as bridge between the leaves
                        interval = ContinuousSet(leaf.path[variable].lower, distribution.cdf.intervals[0].upper)
                        function_value = 1 / (2 * total_samples * interval.range())
                        distribution._quantile.cdf.insert_convex_fragment_left(interval, function_value)
                        distribution._quantile.cdf.normalize()

                    # if the leaf is not the "highest" in this dimension
                    if leaf.path[variable].upper < float("inf"):
                        # create uniform distribution as bridge between the leaves
                        interval = ContinuousSet(distribution.cdf.intervals[-1].lower, leaf.path[variable].upper)
                        function_value = 1 / (2 * total_samples * interval.range())
                        distribution._quantile.cdf.insert_convex_fragment_right(interval, function_value)
                        distribution._quantile.cdf.normalize()


class JPTLike:
    """This one implements an interface for both sum product joint probability trees.
    To be used to construct a new JPT it is necessary that independent marginals and impurities are implemented
    """

    def __init__(self, variables: List[jpt.variables.Variable], jpts: List[JPT]):
        self.variables = variables
        self.jpts = jpts

    def all_splits(self, variables: List[Variable], as_lists: bool = True) -> VariableMap:
        """ Generate a variable map containing all splits of all trees in this collection.
            The splits are sorted (if lists) and unique for each variable.

            @param variables: The variables to collect the splits for
            @param as_lists: Rather to return a VariableMap of lists or sets
            """

        # construct a map for every remaining variable that contains all splits
        all_splits = VariableMap([(v, set()) for v in variables])

        # for every tree in this operation
        for tree in self.jpts:

            # get the splits of the current tree
            current_splits = tree.all_splits(variables, as_lists=False)

            # merge them into the existing splits
            for variable in variables:
                all_splits[variable] |= current_splits[variable]

        # convert splits from sets to lists if desired
        if as_lists:
            for variable, splits in all_splits.items():
                all_splits[variable] = list(splits)

        return all_splits

    def independent_marginals(self, variables: List[jpt.variables.Variable], evidence: jpt.variables.VariableMap,
                              fail_on_unsatisfiability=True) -> PosteriorResult or None:
        """ Compute the marginal distribution of every varialbe in 'variables' assuming independence.
        Unlike JPT.posterior, this method also can compute marginals on variables that are in the evidence.

        :param variables:        the query variables of the posterior to be computed
        :type variables:         list of jpt.variables.Variable
        :param evidence:    the evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :type fail_on_unsatisfiability:  bool
        :return:            jpt.trees.InferenceResult containing distributions, candidates and weights
        """
        raise NotImplementedError("This is an abstract class.")


class SumJPT(JPTLike):
    """ Represent a sum of JPTs that can be used as a training basis for new JPTs.
        This is needed in the variable nodes of factor graphs.
     """

    def independent_marginals(self, variables: List[jpt.variables.Variable], evidence: jpt.variables.VariableMap,
                              fail_on_unsatisfiability=True) -> PosteriorResult or None:

        # construct a list with the independent marginals
        independent_marginals = []

        # collect marginals
        for tree in self.jpts:
            marginals = tree.independent_marginals(variables, evidence, False)
            if marginals is not None:
                independent_marginals.append(marginals)

        # return None if there are no valid distributions
        if len(independent_marginals) == 0:
            return None

        # calculate weight sum
        weight_sum = sum(r.result for r in independent_marginals)

        # if everything has 0 probability this query is impossible
        if weight_sum == 0:
            return None

        # construct result type
        result = jpt.trees.PosteriorResult(independent_marginals[0].query, independent_marginals[0].evidence,
                                           dists=VariableMap())
        result.result = sum(r.result for r in independent_marginals) / len(independent_marginals)

        # normalize weights
        weights = [r.result/weight_sum for r in independent_marginals]

        # for every variable
        for variable in variables:

            # merge (form the union) of every variable's distributions independently
            if variable.numeric:
                result.distributions[variable] = Numeric.merge([r.distributions[variable]
                                                                for r in independent_marginals],
                                                               weights=weights)
            elif variable.symbolic:
                result.distributions[variable] = Multinomial.merge([r.distributions[variable]
                                                                    for r in independent_marginals],
                                                                   weights=weights)

        return result


class ProductJPT(JPTLike):
    """ Represent a product of JPTs that can be used as a training basis for new JPTs.
        This is needed in the factor nodes of factor graphs.
     """

    def independent_marginals(self, variables: List[jpt.variables.Variable], evidence: jpt.variables.VariableMap,
                              fail_on_unsatisfiability=True) -> PosteriorResult or None:

        # construct a list with the independent marginals
        independent_marginals = []

        # collect marginals
        for tree in self.jpts:
            marginals = tree.independent_marginals(variables, evidence, False)
            if marginals is not None:
                independent_marginals.append(marginals)
            else:
                # in the product only one marginal needs to be invalid for the whole product to become invalid
                return None

        # construct the result type
        result = jpt.trees.PosteriorResult(independent_marginals[0].query, independent_marginals[0].evidence,
                                           dists=VariableMap())
        result.result = sum(r.result for r in independent_marginals) / len(independent_marginals)

        # for every variable
        for variable in variables:

            # form the intersection of every variable's distributions independently
            if variable.numeric:
                result.distributions[variable] = Numeric.product([r.distributions[variable]
                                                                  for r in independent_marginals],
                                                                 weights=[r.result for r in independent_marginals])
            elif variable.symbolic:
                result.distributions[variable] = Multinomial.product([r.distributions[variable]
                                                                      for r in independent_marginals],
                                                                     weights=[r.result for r in independent_marginals])
        return result
