"""© Copyright 2021, Mareike Picklum, Daniel Nyga."""
import datetime
import html
import json
import math
import numbers
import os
import pickle
from collections import defaultdict, deque, ChainMap, OrderedDict
from itertools import zip_longest
from operator import attrgetter
from typing import Dict, List, Tuple, Any, Union, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from dnutils import first, ifnone, mapstr, err, fst, out, ifnot, getlogger, logs
from graphviz import Digraph
from matplotlib import style, pyplot as plt

from .base.constants import plotstyle, orange, green
from .base.errors import Unsatisfiability
from .base.utils import list2interval, format_path, normalized
from .base.utils import prod, setstr_int
from .distributions import Integer
from .distributions import Multinomial, Numeric
from .variables import VariableMap, SymbolicVariable, NumericVariable, Variable, VariableAssignment, IntegerVariable, \
    LabelAssignment, ValueAssignment

try:
    from .base.intervals import __module__
    from .learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet, RealSet
    from .learning.impurity import Impurity


style.use(plotstyle)


# ----------------------------------------------------------------------------------------------------------------------
# Global constants
DISCRIMINATIVE = 'discriminative'
GENERATIVE = 'generative'


# ----------------------------------------------------------------------------------------------------------------------


class Node:
    """
    Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.
    """

    def __init__(self, idx: int, parent: None or 'DecisionNode' = None) -> None:
        """
        Create a Node
        :param idx: the identifier of a node
        :param parent: the parent of this node
        """
        self.idx = idx
        self.parent: DecisionNode = parent
        self.samples = 0.
        self._path = []

    @property
    def path(self) -> VariableMap:
        """
        :return: the path of this Node as VariableMap
        """
        res = VariableMap()
        for var, vals in self._path:
            res[var] = (res.get(
                var,
                set(range(var.domain.n_values)) if (var.symbolic or var.integer) else R
            ).intersection(vals))
        return res

    def consistent_with(self, evidence: VariableMap) -> bool:
        """
        Check if the node is consistent with the variable assignments in evidence.

        :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :return: bool
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
                        if value.isdisjoint(restriction):
                            return False

                    # if it is a singular value
                    else:
                        # check if the path allows this value
                        if not restriction.lower < value <= restriction.upper:
                            return False

                # if the variable is symbolic or integer
                elif variable.symbolic or variable.integer:

                    # if it is a set of possible values
                    if not isinstance(value, set):
                        value = set([value])
                    # check if the sets intersect
                    if value.isdisjoint(restriction):
                        return False

        return True

    def format_path(self):
        return format_path(self.path)

    def number_of_parameters(self) -> int:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'Node<{self.idx}>'

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'

    def depth(self) -> int:
        """
        :return: the depth of this node
        """
        return len(self._path)

    def contains(self, samples: np.ndarray, variable_index_map: VariableMap) -> np.array:
        """
        Check if this node contains the given samples in parallel.

        :param samples: The samples to check
        :param variable_index_map: A VariableMap mapping to the indices in 'samples'
        :return: numpy array with 0s and 1s
        """
        result = np.ones(len(samples))
        for variable, restriction in self.path.items():
            index = variable_index_map[variable]
            if variable.numeric:
                result *= (samples[:, index] > restriction.lower) & (samples[:, index] <= restriction.upper)
            if variable.symbolic or variable.integer:
                result *= np.isin(samples[:, index], list(restriction))

        return result


# ----------------------------------------------------------------------------------------------------------------------

class DecisionNode(Node):
    """
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    """

    def __init__(self, idx: int, variable: Variable, parent: 'DecisionNode' or None = None):
        """
        Create a DecisionNode

        :param idx: The identifier of a node
        :param variable: The split variable
        :param parent: The parent of this node
        """
        self._splits = None
        self.variable = variable
        super().__init__(idx, parent=parent)
        self.children: None or List[Node] = None  # [None] * len(self.splits)

    def __eq__(self, o) -> bool:
        return (
            type(self) is type(o) and
            self.idx == o.idx and
            (self.parent.idx
             if self.parent is not None else None) == (o.parent.idx if o.parent is not None else None) and
            [n.idx for n in self.children] == [n.idx for n in o.children] and
            self.splits == o.splits and
            self.variable == o.variable and
            self.samples == o.samples
        )

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The DecisionNode as a json serializable dict.
        """
        return {
            'idx': self.idx,
            'parent': ifnone(self.parent, None, attrgetter('idx')),
            'splits': [
                s.to_json() if isinstance(s, ContinuousSet) else list(s)
                for s in self.splits
            ],
            'variable': self.variable.name,
            '_path': [
                (var.name, split.to_json() if var.numeric else list(split))
                for var, split in self._path
            ],
            'children': [node.idx for node in self.children],
            'samples': self.samples,
            'child_idx': self.parent.children.index(self) if self.parent is not None else None
        }

    @staticmethod
    def from_json(tree: 'JPT', data: Dict[str, Any]) -> 'DecisionNode':
        """
        Construct a Decision node from a json dict.
        :param tree: The tree to mount the node in
        :param data: The data describing the members of the node
        :return: the constructed and mounted DecisionNode
        """
        node = DecisionNode(
            idx=data['idx'],
            variable=tree.varnames[data['variable']]
        )
        node.splits = [
            Interval.from_json(s) if node.variable.numeric else set(s)
            for s in data['splits']
        ]
        node.children = [None] * len(node.splits)
        node.parent = ifnone(data['parent'], None, tree.innernodes.get)
        node.samples = data['samples']
        if node.parent is not None:
            node.parent.set_child(data['child_idx'], node)
        tree.innernodes[node.idx] = node
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
        """
        Set the child at ``index`` of this Node. Also extend the path of the child node with this
        nodes' path.
        :param idx: the idx of the child (0 for left, 1 for right)
        :param node: The child
        """
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.variable, self.splits[idx]))

    def str_edge(self, idx_split: int) -> str:
        """
        Convert the edge to child at ``idx`` to a string.
        :param idx_split: The index of the child
        :return: str
        """
        if self.variable.numeric:
            return self.variable.str(
                self.splits[idx_split],
                fmt='logic'
            )
        elif self.variable.symbolic:
            negate = len(self.splits[1]) > 1
            if negate:
                label = self.variable.domain.labels[fst(self.splits[0])]
                return '%s%s' % ('\u00AC' if idx_split > 0 else '', label)
            else:
                return str(self.variable.domain.labels[fst(self.splits[idx_split])])
        elif self.variable.integer:
            return setstr_int(self.variable.domain.value2label(self.splits[idx_split]))

    @property
    def str_node(self) -> str:
        return self.variable.name

    def recursive_children(self):
        """
        :return: All children of this node
        """
        return self.children + [item for sublist in
                                [child.recursive_children() for child in self.children] for item in sublist]

    def __str__(self) -> str:
        return (f'<DecisionNode #{self.idx} '
                f'{self.variable.name} = [%s]' % '; '.join(self.str_edge(i) for i in range(len(self.splits))) +
                f'; parent-#: {self.parent.idx if self.parent is not None else None}'
                f'; #children: {len(self.children)}>')

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                 2 are parameters necessary since it the variable and its splitting value
                 are sufficient to describe this computation unit.
        """
        return 2


# ----------------------------------------------------------------------------------------------------------------------

class Leaf(Node):
    """
    Represents a leaf node of the :class:`jpt.trees.Tree`.
    """

    def __init__(self, idx: int, parent: DecisionNode or None = None, prior: float or None = None):
        """
        Construct a Leaf
        :param idx: the index of this leaf
        :param parent: the parent of this leaf
        :param prior: the prior of this leaf (relative number of samples in this leaf)
        """
        super().__init__(idx, parent=parent)
        self.distributions = VariableMap()
        self.prior = prior
        self.s_indices = []

    @property
    def str_node(self) -> str:
        return ""

    def applies(self, query: VariableAssignment) -> bool:
        """
        Checks whether this leaf is consistent with the given ``query``.
        :param query: the query to check
        :return: bool
        """
        if isinstance(query, LabelAssignment):
            query = query.value_assignment()
        path = self.path
        for var in set(query.keys()).intersection(set(path.keys())):
            if path.get(var).isdisjoint(query.get(var)):
                return False
        return True

    @property
    def value(self):
        return self.distributions

    def recursive_children(self):
        """
        :return: All children of this node
        """
        return []

    def __str__(self) -> str:
        return f'<Leaf #{self.idx}; parent: #%s prior = %.3f>' % (
            ifnone(self.parent, None, attrgetter('idx')),
            self.prior
        )

    def __repr__(self) -> str:
        return f'Leaf<{self.idx}> object at {hex(id(self))}'

    def __hash__(self):
        return hash((type(self), ((k.name, v) for k, v in self.distributions.items()), self.prior))

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The DecisionNode as a json serializable dict.
        """
        return {
            'idx': self.idx,
            'distributions': self.distributions.to_json(),
            'prior': self.prior,
            'samples': self.samples,
            's_indices': [int(i) for i in self.s_indices],
            'parent': ifnone(self.parent, None, attrgetter('idx')),
            'child_idx': self.parent.children.index(self) if self.parent is not None else -1
        }

    @staticmethod
    def from_json(tree: 'JPT', data: Dict[str, Any]) -> 'Leaf':
        """
        Construct a Decision node from a json dict.
        :param tree: The tree to mount the node in
        :param data: The data describing the members of the node
        :return: the constructed and mounted DecisionNode
        """
        leaf = Leaf(
            idx=data['idx'],
            prior=data['prior'],
            parent=tree.innernodes.get(data['parent'])
        )
        leaf.distributions = VariableMap(
            {
                tree.varnames[v]: tree.varnames[v].domain.from_json(d) for v, d in data['distributions'].items()
            }
        )
        leaf._path = []
        if leaf.parent is not None:
            leaf.parent.set_child(data['child_idx'], leaf)
        leaf.prior = data['prior']
        leaf.samples = data['samples']
        if 's_indices' in data:
            leaf.s_indices = np.array(data['s_indices'])
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

        :param evidence: A preprocessed VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        """
        return self.probability(evidence) > 0.

    def path_consistent_with(self, evidence: VariableMap) -> bool:
        """
        Check if the path of this node is consistent with the variable assignments in evidence.

        :param evidence: A preprocessed VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        """
        return super(Leaf, self).consistent_with(evidence)

    def probability(self,
                    query: VariableAssignment,
                    dirac_scaling: float = 2.,
                    min_distances: VariableMap = None) -> float:
        """
        Calculate the probability of a (partial) query. Exploits the independence assumption
        :param query: A preprocessed VariableMap that maps to singular values (numeric or symbolic)
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
        if isinstance(query, LabelAssignment):
            query = query.value_assignment()
        # for every variable and its assignment
        for variable, value in query.items():
            variable: Variable

            # if it is a numeric
            if variable.numeric:
                result *= self._numeric_probability(variable, value, dirac_scaling, min_distances)

            # if the variable is symbolic
            elif variable.symbolic or variable.integer:

                # force the evidence to be a set
                if not isinstance(value, set):
                    value = set([value])

                # return false if the evidence is impossible in this leaf
                result *= self.distributions[variable]._p(value)

        return result

    def _numeric_probability(self, variable: NumericVariable, value, dirac_scaling: float = 2.,
                             min_distances: VariableMap = None):
        """ Calculate the probability of an arbitrary value for a numeric variable.
        :param variable: A numeric variable
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        """
        if isinstance(value, RealSet):
            return sum(self._numeric_probability(variable, cs, dirac_scaling, min_distances) for cs in value.intervals)

        # handle ContinuousSet
        elif isinstance(value, ContinuousSet):

            if value.size() == 0:
                return 0

            elif value.size() == 1:
                return self._numeric_probability(variable, value.lower, dirac_scaling, min_distances)

            else:
                return self.distributions[variable]._p(value)

        # handle single Numbers
        elif isinstance(value, numbers.Number):
            result = 1.
            # get the likelihood
            likelihood = self.distributions[variable].pdf(value)

            # if it is infinity and no handling is provided replace it with 1.
            if likelihood == float("inf") and not min_distances:
                result *= 1
            # if it is infinite and a handling is provided, replace with dirac_sclaing/min_distance
            elif likelihood == float("inf") and min_distances:
                result *= dirac_scaling / min_distances[variable]
            else:
                result *= likelihood

            return result

        else:
            raise ValueError("Unknown Datatype for Conditional JPT, type is %s" % type(value))

    def parallel_likelihood(self, queries: np.ndarray, dirac_scaling: float = 2.,  min_distances: VariableMap = None) \
            -> np.ndarray:
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
            if isinstance(variable, SymbolicVariable) or isinstance(variable, IntegerVariable):

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

            else:
                raise ValueError("Variable of type %s is not known!" % type(variable))

            # multiply results
            result *= probs

        return result

    def mpe(self, minimal_distances: VariableMap) -> (float, VariableMap):
        """
        Calculate the most probable explanation of this leaf as a fully factorized distribution.
        :return: the likelihood of the maximum as a float and the configuration as a VariableMap
        """

        # initialize likelihood and maximum
        result_likelihood = self.prior
        maximum = dict()

        # for every variable and distribution
        for variable, distribution in self.distributions.items():

            # calculate mpe of that distribution
            likelihood, explanation = distribution.mpe()

            # apply upper cap for infinities
            likelihood = minimal_distances[variable] if likelihood == float("inf") else likelihood

            # update likelihood
            result_likelihood *= likelihood

            # save result
            maximum[variable] = explanation

        # create mpe result
        return result_likelihood, LabelAssignment(maximum.items())

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                Leafs require 1 + the sum of all distributions parameters. The 1 extra parameter
                represents the prior.
        """
        return sum([distribution.number_of_parameters() for distribution in self.distributions.values()])

    def sample(self, amount) -> np.array:
        """Sample `amount` many samples from the leaf.

        :returns A numpy array of size (amount, self.variables) containing the samples.
        """
        result = np.empty((amount, len(self.distributions)), dtype=object)

        for idx, (variable, distribution) in enumerate(self.distributions.items()):
            result[:, idx] = list(distribution.sample(amount))

        return result


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyProtectedMember
class JPT:
    """
    Implementation Joint Probability Trees (JPTs).
    """

    logger = getlogger('/jpt', level=logs.INFO)

    def __init__(self,
                 variables: List[Variable],
                 targets: List[str or Variable] = [],
                 features: List[str or Variable] = [],
                 min_samples_leaf: float or int = .01,
                 min_impurity_improvement: float or None = None,
                 max_leaves: int or None = None,
                 max_depth: int or None = None,
                 dependencies: Dict[Variable, List[Variable]] or None = None) -> None:
        """
        Create a JPT.
        :param variables: The variables that will be represented this model
        :param targets: The variables where the information gain will be computed on
        :param features: The variables where the splits will be chosen from
        :param min_samples_leaf: If an integer is provided it is the minimal number of samples required to form a leaf,
        if a float is provided it will be the minimal percentage of samples that is required to form a leaf
        :param min_impurity_improvement: The minimal amount of information gain to justify a split
        :param max_leaves: The maximum number of leaves (deprecated)
        :param max_depth: The maximum depth the tree may have
        :param dependencies: A dictionary mapping variables to a list of dependent variables. Having this
        sparse may speed up training a lot.
        """

        self._variables = list(variables)
        self.varnames: OrderedDict[str, Variable] = OrderedDict((var.name, var) for var in self._variables)
        self._targets = (
            list(self.variables)
            if not targets else [self.varnames[v] if type(v) is str else v for v in targets]
        )

        # handle features such that only specifying targets is enough
        if not targets:
            if not features:
                self._features = list(self.variables)
            else:
                self._features = [self.varnames[v] if type(v) is str else v for v in features]
        else:
            if not features:
                self._features = [v for v in self.variables if v not in self.targets]
            else:
                self._features = [self.varnames[v] if type(v) is str else v for v in features]

        self.leaves: Dict[int, Leaf] = {}
        self.innernodes: Dict[int, DecisionNode] = {}
        self.priors: VariableMap = VariableMap()

        self._min_samples_leaf = min_samples_leaf
        self._keep_samples = False
        self.min_impurity_improvement = ifnone(min_impurity_improvement, 0)

        # a map saving the minimal distances to prevent infinite high likelihoods
        self.minimal_distances: VariableMap = VariableMap(variables=self.variables)
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
        if dependencies is None:
            self.dependencies: VariableMap[Variable, List[Variable]] = VariableMap({
                var: list(self.targets) for var in self.features
            })
        else:
            self.dependencies: VariableMap[Variable, List[Variable]] = VariableMap(
                dependencies.items(),
                variables=self.variables
            )

    def _reset(self) -> None:
        """ Delete all parameters of this model (not the hyperparameters)"""
        self.innernodes.clear()
        self.leaves.clear()
        self.priors = VariableMap(variables=self.variables) # .clear()
        self.root = None
        self.c45queue.clear()

    @property
    def allnodes(self):
        return ChainMap(self.innernodes, self.leaves)

    @property
    def variables(self) -> Tuple[Variable]:
        return self._variables

    @property
    def targets(self) -> List[Variable]:
        return self._targets

    @property
    def features(self) -> List[Variable]:
        return self._features

    @property
    def numeric_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, NumericVariable)]

    @property
    def symbolic_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, SymbolicVariable)]

    @property
    def integer_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, IntegerVariable)]

    @property
    def numeric_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, NumericVariable)]

    @property
    def symbolic_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, SymbolicVariable)]

    @property
    def integer_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, IntegerVariable)]

    @property
    def numeric_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, NumericVariable)]

    @property
    def symbolic_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, SymbolicVariable)]

    @property
    def integer_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, IntegerVariable)]

    def to_json(self) -> Dict[str, Any]:
        """Convert the tree to a json dictionary that can be serialized. """
        return {
            'variables': [v.to_json() for v in self.variables],
            'targets': [v.name for v in self.targets] if self.targets else self.targets,
            'features': [v.name for v in self.features],
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity_improvement': self.min_impurity_improvement,
            'max_leaves': self.max_leaves,
            'max_depth': self.max_depth,
            'minimal_distances': self.minimal_distances.to_json(),
            'dependencies': {
                var.name: [v.name for v in deps]
                for var, deps in self.dependencies.items()},
            'leaves': [l.to_json() for l in self.leaves.values()],
            'innernodes': [n.to_json() for n in self.innernodes.values()],
            'priors': {variable.name: p.to_json() for variable, p in self.priors.items()},
            'root': ifnone(self.root, None, attrgetter('idx'))
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'JPT':
        """Construct a tree from a json dict."""
        variables = OrderedDict([(d['name'], Variable.from_json(d)) for d in data['variables']])
        jpt = JPT(
            variables=list(variables.values()),
            targets=(
                [variables[v] for v in data['targets']]
                if data.get('targets')
                else []
            ),
            features=(
                [variables[v] for v in data['features']]
                if data.get('features')
                else []
            ),
            min_samples_leaf=data['min_samples_leaf'],
            min_impurity_improvement=data['min_impurity_improvement'],
            max_leaves=data['max_leaves'],
            max_depth=data['max_depth'],
            dependencies={
                variables[var]: [variables[v] for v in deps]
                for var, deps in data.get('dependencies', {}).items()
            }
        )
        jpt.minimal_distances = VariableMap.from_json(
            jpt.numeric_variables,
            data.get("minimal_distances", {})
        )
        for d in data['innernodes']:
            DecisionNode.from_json(jpt, d)
        for d in data['leaves']:
            Leaf.from_json(jpt, d)

        jpt.priors = VariableMap({
            jpt.varnames[varname]: jpt.varnames[varname].domain.from_json(dist)
            for varname, dist in data['priors'].items()
        })
        jpt.root = jpt.allnodes[data.get('root')] if data.get('root') is not None else None
        return jpt

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = JPT.from_json(state).__dict__

    def __eq__(self, o) -> bool:
        return all((
            isinstance(o, JPT),
            self.innernodes == o.innernodes,
            self.leaves == o.leaves,
            self.priors == o.priors,
            self.min_samples_leaf == o.min_samples_leaf,
            self.min_impurity_improvement == o.min_impurity_improvement,
            self.targets == o.targets,
            self.variables == o.variables,
            self.max_depth == o.max_depth,
            self.max_leaves == o.max_leaves,
            self.dependencies == o.dependencies
        ))

    def encode(self, samples: np.ndarray) -> np.array:
        """
        Get the leaf index that describes the partition of each sample. Only works for fully initialized samples, i. e.
        a matrix of arbitrary many rows but #variables many columns.
        :param samples: the samples to evaluate
        :return: A 1D numpy array of integers containing the leaf index of every sample.
        """
        result = np.zeros(len(samples))
        variable_index_map = VariableMap([(variable, idx) for (idx, variable) in enumerate(self.variables)])
        samples = self._preprocess_data(samples)
        for idx, leaf in self.leaves.items():
            contains = leaf.contains(samples, variable_index_map)
            result[contains == 1] = idx
        return result

    def pdf(self, values: VariableAssignment) -> float:
        """
        Get the likelihood of one world
        :param values: A VariableMap mapping some variables to one value.
        :return: The likelihood as float
        """
        if isinstance(values, LabelAssignment):
            values = values.value_assignment()
        values_ = ValueAssignment([(var, ContinuousSet(val, val)) for var, val in values.items()])
        pdf = 0
        for leaf in self.apply(values_):
            pdf += leaf.prior * (prod(leaf.distributions[var].pdf(value)
                                      for var, value in values.items()) if values else 1)
        return pdf

    def infer(self,
              query: Union[Dict[Union[Variable, str], Any], VariableAssignment],
              evidence: Union[Dict[Union[Variable, str], Any], VariableAssignment] = None,
              fail_on_unsatisfiability: bool = True) -> float or None:
        r"""For each candidate leaf ``l`` calculate the number of samples in which `query` is true:

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
        :param fail_on_unsatisfiability: whether an error is raised in case of unsatisfiable evidence or not.
        """
        if isinstance(query, dict):
            query = self.bind(query)
        if isinstance(query, LabelAssignment):
            query_ = query.value_assignment()
        else:
            query_ = query
        if evidence is None or isinstance(evidence, dict):
            evidence = self.bind(evidence)
        if isinstance(evidence, LabelAssignment):
            evidence_ = evidence.value_assignment()
        else:
            evidence_ = evidence

        p_q = 0.
        p_e = 0.

        for leaf in self.apply(evidence_):
            p_m = 1
            likelihoods = []
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var in leaf.path:  # var.numeric and
                    evidence_val = evidence_val.intersection(leaf.path[var])
                # elif (var.symbolic or var.integer) and var in leaf.path:
                #     continue
                p_m *= leaf.distributions[var]._p(evidence_val)
                likelihoods.append((var, evidence_val, leaf.distributions[var]._p(evidence_val)))

            w = leaf.prior
            p_m *= w
            p_e += p_m

            if leaf.applies(query_):
                for var in set(query_.keys()):
                    query_val = query_[var]
                    if var.numeric and var in leaf.path:
                        query_val = query_val.intersection(leaf.path[var])
                    elif (var.symbolic or var.integer) and var in leaf.path:
                        continue
                    p_m *= leaf.distributions[var]._p(query_val)
                p_q += p_m


        if p_e == 0:
            if fail_on_unsatisfiability:
                raise ValueError('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence))
            else:
                return None
        else:
            return p_q/p_e

    # noinspection PyProtectedMember
    def posterior(self,
                  variables: List[Variable or str] = None,
                  evidence: Dict[Union[Variable, str], Any] or VariableAssignment = None,
                  fail_on_unsatisfiability: bool = True,
                  report_inconsistencies: bool = False) -> VariableMap or None:
        """
        Compute the posterior distribution of every variable in ``variables``. The result contains independent
        distributions. Be aware that they might not actually be independent.

        :param variables: The query variables of the posterior to be computed
        :param evidence: The evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: Rather or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :param report_inconsistencies:   In case of an ``Unsatisfiability`` error, the exception raise
                                         will contain information about the variable assignments that
                                         caused the inconsistency.
        :return: jpt.trees.PosteriorResult containing distributions, candidates and weights
        """
        if isinstance(evidence, dict):
            evidence = self.bind(evidence)
        if isinstance(evidence, LabelAssignment):
            evidence_ = evidence.value_assignment()
        else:
            evidence_ = ifnone(evidence, {})

        variables = ifnone(variables, self.variables)
        result = VariableMap()
        variables = [self.varnames[v] if type(v) is str else v for v in variables]

        distributions = defaultdict(list)

        likelihoods = []
        priors = []

        inconsistencies = {}
        for leaf in self.apply(evidence_):
            likelihood = 1
            conflicting_assignment = VariableMap()
            # check if path of candidate leaf is consistent with evidence
            # (i.e. contains evicence variable with *correct* value or does not contain it at all)
            for var in set(evidence_.keys()):
                evidence_set = evidence_[var]
                if var in leaf.path:
                    evidence_set = evidence_set.intersection(leaf.path[var])

                if isinstance(evidence_set, ContinuousSet) and evidence_set.size() == 1:
                    l_var = leaf.distributions[var].pdf(evidence_set.lower)
                    l_var = 1 if np.isinf(l_var) else l_var
                else:
                    l_var = leaf.distributions[var]._p(evidence_set)

                if not l_var:
                    conflicting_assignment[var] = var.domain.value2label(evidence_set)
                    if not report_inconsistencies:
                        break

                likelihood *= ifnot(l_var, 1)

            if conflicting_assignment:
                inconsistencies[conflicting_assignment] = inconsistencies.get(conflicting_assignment, 0) + likelihood
                continue

            likelihoods.append(0 if conflicting_assignment else likelihood)
            priors.append(leaf.prior)

            for var in variables:
                evidence_set = evidence_.get(var)
                distribution = leaf.distributions[var]
                if evidence_set is not None:
                    if var in leaf.path:
                        evidence_set = evidence_set.intersection(leaf.path[var])
                        distribution = distribution.crop(evidence_set)
                distributions[var].append(distribution)

        weights = [l * p for l, p in zip(likelihoods, priors)]
        try:
            weights = normalized(weights)
        except ValueError:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Evidence %s is unsatisfiable.' % format_path(evidence),
                                       reasons=inconsistencies)
            return None


        for var, dists in distributions.items():
            if var.numeric:
                result[var] = Numeric.merge(dists, weights=weights)
            elif var.symbolic:
                result[var] = Multinomial.merge(dists, weights=weights)
            elif var.integer:
                result[var] = Integer.merge(dists, weights=weights)

        return result

    def expectation(self,
                    variables: Iterable[Variable] = None,
                    evidence: VariableAssignment = None,
                    fail_on_unsatisfiability: bool = True) -> VariableMap or None:
        """
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.

        :param variables: The variables to compute the expectation distributions on
        :param evidence: The raw evidence applied to the tree
        :param fail_on_unsatisfiability: Rather or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :return: VariableMap
        """
        if evidence is None:
            evidence = self.bind()
        if isinstance(evidence, LabelAssignment):
            evidence_ = evidence.value_assignment()
        else:
            evidence_ = evidence

        variables = ifnot(
            [v if isinstance(v, Variable) else self.varnames[v] for v in ifnone(variables, self.variables)],
            set(self.variables) - set(evidence_ or {})
        )

        posteriors = self.posterior(
            variables,
            evidence_,
            fail_on_unsatisfiability=fail_on_unsatisfiability
        )

        if posteriors is None:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence))
            else:
                return None

        final = VariableMap()
        for var, dist in posteriors.items():
            final[var] = dist.expectation()
        return final

    def mpe(self, evidence: Union[Dict[Union[Variable, str], Any], VariableAssignment] = None,
            fail_on_unsatisfiability: bool = True) -> (List[LabelAssignment], float) or None:
        """
        Calculate the most probable explanation of all variables if the tree given the evidence.
        :param evidence: The evidence that is applied to the tree
        :param fail_on_unsatisfiability: Rather or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :return: List of LabelAssignments that describes all maxima of the tree given the evidence.
            Additionally, a float describing the likelihood of all solutions is returned.
        """
        if isinstance(evidence, LabelAssignment):
            evidence_ = evidence.value_assignment()
        else:
            evidence_ = evidence

        # apply the conditions given
        conditional_jpt = self.conditional_jpt(evidence_, fail_on_unsatisfiability)

        if conditional_jpt is None:
            return None

        # calculate the maximal probabilities for each leaf
        maxima = [leaf.mpe(self.minimal_distances) for leaf in conditional_jpt.leaves.values()]

        # get the maximum of those maxima
        highest_likelihood = max([m[0] for m in maxima])

        # create a list for all possible maximal occurrences
        results = []

        # for every leaf and its mpe
        for leaf, (likelihood, mpe) in zip(conditional_jpt.leaves.values(), maxima):

            if likelihood == highest_likelihood:
                # append the argmax to the results
                results.append(mpe)

        # return the results
        return results, highest_likelihood

    def _preprocess_query(self,
                          query: Union[dict, VariableMap],
                          remove_none: bool = True,
                          skip_unknown_variables: bool = False,
                          allow_singular_values: bool = False) -> LabelAssignment:
        """
        Transform a query entered by a user into an internal representation that can be further processed.
        :param query: the raw query
        :param remove_none: Rather to remove None entries or not
        :param skip_unknown_variables:  skip preprocessing for variable that does not exist in tree (may happen in
                                        multiple reverse tree inference). If False, an exception is raised;
                                        default: False
        :param allow_singular_values: Allow singular values, such that they are transformed to the daomain
            specification of numeric variables but not transformed to intervals via the PPF.
        :return: the preprocessed VariableMap
        """
        # Transform lists into a numeric interval:
        query_ = LabelAssignment(variables=self.variables)
        # parameter of the respective variable:
        for key, arg in query.items():
            if arg is None and remove_none:
                continue

            var = key if isinstance(key, Variable) else self.varnames.get(key)

            if var is None:
                if skip_unknown_variables:
                    continue
                else:
                    raise Exception(f'Variable "{key}" is unknown!')

            if var.numeric:
                if type(arg) is list:
                    arg = list2interval(arg)
                if isinstance(arg, numbers.Number):
                    val = arg
                    if allow_singular_values:
                        query_[var] = val
                    # Apply a "blur" to single value evidences, if any blur is set
                    elif var.blur:
                        prior = self.priors[var.name]
                        quantile = prior.cdf.functions[
                            max(1, min(len(prior.cdf) - 2, prior.cdf.idx_at(var.domain.label2value(val))))
                        ].eval(val)
                        lower = var.domain.label2value(quantile - var.blur / 2)
                        upper = var.domain.label2value(quantile + var.blur / 2)
                        query_[var] = ContinuousSet(
                            var.domain.value2label(
                                prior.ppf.functions[max(1, min(len(prior.cdf) - 2, prior.ppf.idx_at(lower)))].eval(
                                    lower
                                )
                            ),
                            var.domain.value2label(
                                prior.ppf.functions[min(len(prior.ppf) - 2, max(1, prior.ppf.idx_at(upper)))].eval(
                                    upper
                                )
                            )
                        )
                    else:
                        query_[var] = ContinuousSet(val, val)
                elif isinstance(arg, ContinuousSet):
                    query_[var] = arg
                elif isinstance(arg, RealSet):
                    query_[var] = RealSet([
                        ContinuousSet(i.lower, i.upper, i.left, i.right) for i in arg.intervals
                    ])
                else:
                    raise TypeError('Unknown type of variable value: %s' % type(arg).__name__)
            if var.symbolic or var.integer:
                # Transform into internal values (symbolic values to their indices):
                if type(arg) is list:
                    arg = var.domain.list2set(arg)
                if type(arg) is tuple:
                    raise TypeError('Illegal type for values of domain %s: %s'
                                    % (var.domain.__name__, type(arg).__name__))
                if type(arg) is not set:
                    arg = {arg}
                query_[var] = {v for v in arg}

        return query_

    def apply(self, query: VariableAssignment) -> Iterator[Leaf]:
        """
        Iterator that yields leaves that are consistent with ``query``.
        :param query: the preprocessed query
        :return:
        """
        if isinstance(query, LabelAssignment):
            query = query.value_assignment()
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

    def c45(self,
            data: np.ndarray,
            start: int,
            end: int,
            parent: DecisionNode,
            child_idx: int,
            depth: int) -> None:
        """
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.

        :param data:        the indices for the training samples used to calculate the gain.
        :param start:       the starting index in the data.
        :param end:         the stopping index in the data.
        :param parent:      the parent node of the current iteration, initially ``None``.
        :param child_idx:   the index of the child in the current iteration.
        :param depth:       the depth of the tree in the current recursion level.
        """
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = ifnone(self.min_impurity_improvement, 0)
        n_samples = end - start
        split_var_idx = split_pos = -1
        split_var = None
        impurity = self.impurity

        max_gain = impurity.compute_best_split(start, end)
        if max_gain < 0:
            raise ValueError('Something went wrong!')

        self.logger.debug('Data range: %d-%d,' % (start, end),
                          'split var:', split_var,
                          ', split_pos:', split_pos,
                          ', gain:', max_gain)

        if max_gain:
            split_pos = impurity.best_split_pos
            split_var_idx = impurity.best_var
            split_var = self.variables[split_var_idx]

        if max_gain <= min_impurity_improvement or depth >= self.max_depth:  # Create a leaf node ----------------------
            leaf = node = Leaf(idx=len(self.allnodes), parent=parent)

            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.distribution()._fit(
                    data=data,
                    rows=self.indices[start:end],
                    col=i
                )
            leaf.prior = n_samples / data.shape[0]
            leaf.samples = n_samples
            if self._keep_samples:
                leaf.s_indices = self.indices[start:end]

            self.leaves[leaf.idx] = leaf

        else:  # Create a decision node --------------------------------------------------------------------------------
            node = DecisionNode(
                idx=len(self.allnodes),
                variable=split_var,
                parent=parent
            )
            node.samples = n_samples
            self.innernodes[node.idx] = node

            if split_var.symbolic:  # Symbolic domain ------------------------------------------------------------------
                split_value = int(data[self.indices[start + split_pos], split_var_idx])
                splits = [
                    {split_value},
                    set(split_var.domain.values.values()) - {split_value}
                ]

            elif split_var.numeric:  # Numeric domain ------------------------------------------------------------------
                split_value = (data[self.indices[start + split_pos], split_var_idx] +
                               data[self.indices[start + split_pos + 1], split_var_idx]) / 2
                splits = [
                    Interval(np.NINF, split_value, EXC, EXC),
                    Interval(split_value, np.PINF, INC, EXC)
                ]

            elif split_var.integer:  # Integer domain ------------------------------------------------------------------
                split_value = int(data[self.indices[start + split_pos + 1], split_var_idx])
                domain = list(split_var.domain.values.values())
                idx_split = domain.index(split_value)
                splits = [set(domain[:idx_split]), set(domain[idx_split:])]

            else:  # ---------------------------------------------------------------------------------------------------
                raise TypeError('Unknown variable type: %s.' % type(split_var).__name__)

            # recurse left and right
            self.c45queue.append((data, start, start + split_pos + 1, node, 0, depth + 1))
            self.c45queue.append((data, start + split_pos + 1, end, node, 1, depth + 1))

            node.splits = splits

        JPT.logger.debug('Created', str(node))

        if parent is not None:
            parent.set_child(child_idx, node)

        if self.root is None:
            self.root = node

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}\n'
            f'{self.pfmt()}\n'
            f'JPT stats: #innernodes = {len(self.innernodes)}, '
            f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)'
        )

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} '
            f'#innernodes = {len(self.innernodes)}, '
            f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)>'
        )

    def pfmt(self) -> str:
        """
        :return: a pretty-format string representation of this JPT.
        """
        return self._pfmt(self.root, 0)

    def _pfmt(self, node, indent) -> str:
        """
        :param node: The starting node
        :param indent: the indentation of each new level
        :return: a pretty-format string representation of this JPT from node downward.
        """
        return "{}{}\n{}".format(
            " " * indent,
            str(node),
            ''.join([self._pfmt(c, indent + 4) for c in node.children])
            if isinstance(node, DecisionNode) else ''
        )

    def _preprocess_data(self, data=None, rows=None, columns=None) -> np.ndarray:
        """
        Transform the input data into an internal representation.
        :param data: The data to transform
        :param rows: The indices of the rows that will be transformed
        :param columns: The indices of the columns that will be transformed
        :return: the preprocessed data
        """
        if sum(d is not None for d in (data, rows, columns)) > 1:
            raise ValueError('Only either of the three is allowed.')
        elif sum(d is not None for d in (data, rows, columns)) < 1:
            raise ValueError('No data passed.')

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
                raise ValueError(
                    'Unknown variable names: %s'
                    % ', '.join(mapstr(set(self.varnames).symmetric_difference(set(data.columns))))
                )

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

    def learn(self, data=None, rows=None, columns=None, keep_samples=False) -> 'JPT':
        """
        Fit the jpt to ``data``
        :param data:    The training examples (assumed in row-shape)
        :type data:     [[str or float or bool]]; (according to `self.variables`)
        :param rows:    The training examples (assumed in row-shape)
        :type rows:     [[str or float or bool]]; (according to `self.variables`)
        :param columns: The training examples (assumed in column-shape)
        :type columns:  [[str or float or bool]]; (according to `self.variables`)
        :param keep_samples: If true, stores the indices of the original data samples in the leaf nodes. For debugging
                        purposes only. Default is false.
        :return: the fitted model
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check and prepare the data
        _data = self._preprocess_data(data=data, rows=rows, columns=columns)

        for idx, variable in enumerate(self.variables):
            if variable.numeric:
                samples = np.unique(_data[:, idx])
                distances = np.diff(samples)
                self.minimal_distances[variable] = min(distances) if len(distances) > 0 else 2.

        if not _data.shape[0]:
            raise ValueError('No data for learning.')

        self.indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
        self.indices[0] = 0
        np.cumsum(self.indices, out=self.indices)

        JPT.logger.info('Data transformation... %d x %d' % _data.shape)

        # --------------------------------------------------------------------------------------------------------------
        # Initialize the internal data structures
        self._reset()

        # --------------------------------------------------------------------------------------------------------------
        # Determine the prior distributions
        started = datetime.datetime.now()
        JPT.logger.info('Learning prior distributions...')

        for i, (vname, var) in enumerate(self.varnames.items()):
            self.priors[var] = var.distribution()._fit(
                data=_data,
                col=i
            )
        JPT.logger.info(
            '%d prior distributions learnt in %s.' % (
                len(self.priors),
                datetime.datetime.now() - started
            )
        )

        # --------------------------------------------------------------------------------------------------------------
        # Start the training
        if type(self._min_samples_leaf) is int:
            min_samples_leaf = self._min_samples_leaf

        elif type(self._min_samples_leaf) is float and 0 < self._min_samples_leaf < 1:
            min_samples_leaf = max(1, int(self._min_samples_leaf * len(_data)))

        else:
            min_samples_leaf = self._min_samples_leaf

        self._keep_samples = keep_samples

        # Initialize the impurity calculation
        self.impurity = Impurity(self)
        self.impurity.setup(_data, self.indices)
        self.impurity.min_samples_leaf = min_samples_leaf

        started = datetime.datetime.now()
        JPT.logger.info('Started learning of %s x %s at %s '
                        'requiring at least %s samples per leaf' % (_data.shape[0],
                                                                    _data.shape[1],
                                                                    started,
                                                                    min_samples_leaf))
        learning = GENERATIVE if self.targets == self.variables else DISCRIMINATIVE
        JPT.logger.info('Learning is %s. ' % learning)
        if learning == DISCRIMINATIVE:
            JPT.logger.info('Target variables (%d): %s\n'
                            'Feature variables (%d): %s' % (len(self.targets),
                                                            ', '.join(mapstr(self.targets)),
                                                            len(self.variables) - len(self.targets),
                                                            ', '.join(
                                                                mapstr(set(self.variables) - set(self.targets)))))
        # build up tree
        self.c45queue.append((
                _data,
                0,
                _data.shape[0],
                None,
                None,
                0
        ))
        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        # ----------------------------------------------------------------------------------------------------------
        # Print the statistics
        JPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
        JPT.logger.debug(self)
        return self

    fit = learn

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

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

    def likelihood(self,
                   queries: Union[np.ndarray, pd.DataFrame],
                   dirac_scaling: float = 2.,
                   min_distances: Dict = None) -> np.ndarray:
        """
        Get the probabilities of a list of worlds. The worlds must be fully assigned with
        single numbers (no intervals).

        :param queries: An array containing the worlds. The shape is (x, len(variables)).
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :returns: An np.array with shape (x, ) containing the probabilities.
        """

        # set min distances if not overwritten
        if min_distances is None:
            min_distances = self.minimal_distances

        # preprocess the queries
        queries = self._preprocess_data(queries)

        # initialize probabilities
        probabilities = np.zeros(len(queries))

        # for all leaves
        for leaf in self.leaves.values():

            # calculate likelihood
            leaf_probabilities = leaf.parallel_likelihood(
                queries,
                dirac_scaling,
                min_distances
            )

            # multiply likelihood by leaf prior
            probabilities += (leaf.prior * leaf_probabilities)

        return probabilities

    def reverse(self, query, confidence=.05) -> List[Tuple[Dict, List[Node]]]:
        """
        Determines the leaf nodes that match query best and returns their respective paths to the root node.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
        :type query: dict
        :param confidence:  the confidence level for this MPE inference
        :type confidence: float
        :returns: a mapping from probabilities to lists of matcalo.core.algorithms.JPT.Node (path to root)
        :rtype: dict
        """
        # if none of the target variables is present in the query, there is no match possible
        # only check variable names, because multiple trees can have the (semantically) same variable, which differs as
        # python object
        if set([v.name if isinstance(v, Variable) else v for v in query.keys()]).isdisjoint(set(self.varnames)):
            return []

        # Transform into internal values/intervals (symbolic values to their indices)
        query_ = self._preprocess_query(query, skip_unknown_variables=True)

        # update non-query variables to allow all possible values
        for i, var in enumerate(self.variables):
            if var in query_: continue
            if var.numeric:
                query_[var] = R
            else:
                query_[var] = var.domain.values

        # stores the probabilities, that the query variables take on the value(s)/a value in the interval given in
        # the query
        confs = {}

        # find the leaf (or the leaves) that matches the query best
        for k, l in self.leaves.items():
            conf = defaultdict(float)
            for v, dist in l.distributions.items():
                if v.numeric:
                    conf[v] = dist._p(query_[v])
                else:
                    conf_ = 0.
                    for sv in query_[v]:
                        conf_ += dist._p(sv)
                    conf[v] = conf_
            confs[l.idx] = conf

        # the candidates are the leaves that satisfy the confidence requirement (i.e. each free variable of a leaf must satisfy the requirement)
        candidates = sorted([leafidx for leafidx, confs in confs.items() if all(c >= confidence for c in confs.values())],
                            key=lambda l: sum(confs[l].values()), reverse=True)

        out('CANDIDATES in reverse', candidates)

        # for the chosen candidate determine the path to the root
        paths = []
        for c in candidates:
            p = []
            curcand = self.leaves[c]
            while curcand is not None:
                p.append(curcand)
                curcand = curcand.parent
            paths.append((confs[c], p))

        # elements of path are tuples (a, b) with a being mappings of {var: confidence} and b being an ordered list of
        # nodes representing a path from a leaf to the root
        return paths

    def plot(self,
             title: str = "unnamed",
             filename: str or None = None,
             directory: str = '/tmp',
             plotvars: List[Variable] = [],
             view: bool = True,
             max_symb_values: int = 10):
        """
        Generates an SVG representation of the generated regression tree.

        :param title: title of the plot
        :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
        :param directory: the location to save the SVG file to
        :param plotvars: the variables to be plotted in the graph
        :param view: whether the generated SVG file will be opened automatically
        :param max_symb_values: limit the maximum number of symbolic values that are plotted to this number
        """

        plotvars = [self.varnames[v] if type(v) is str else v for v in plotvars]

        if not os.path.exists(directory):
            os.makedirs(directory)

        dot = Digraph(
            format='svg',
            name=title,
            directory=directory,
            filename=f'{filename or title}'
        )

        # create nodes
        sep = ",<BR/>"
        for idx, n in self.leaves.items():
            imgs = ''

            # plot and save distributions for later use in tree plot
            rc = math.ceil(math.sqrt(len(plotvars)))
            img = ''
            for i, pvar in enumerate(plotvars):
                img_name = html.escape(f'{pvar.name}-{n.idx}')

                params = {} if pvar.numeric else {
                    'horizontal': True,
                    'max_values': max_symb_values
                }

                n.distributions[pvar].plot(
                    title=html.escape(pvar.name),
                    fname=img_name,
                    directory=directory,
                    view=False,
                    **params
                )
                img += (f'''{"<TR>" if i % rc == 0 else ""}
                        <TD><IMG SCALE="TRUE" SRC="{os.path.join(directory, f"{img_name}.png")}"/></TD>
                        {"</TR>" if i % rc == rc - 1 or i == len(plotvars) - 1 else ""}
                ''')

                # close current figure to allow for other plots
                plt.close()

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
            title = 'Leaf #%s (p = %.4f)' % (n.idx, n.prior)
            nodelabel = f'''
            <TR>
                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{title}</B><BR/>{html.escape(n.str_node)}</TD>
            </TR>'''

            nodelabel = f'''{nodelabel}{imgs}
                            <TR>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{n.samples} ({n.prior * 100:.3f}%)</TD>
                            </TR>
                            <TR>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{"<B>" + html.escape(v.name) + "</B>" if self.targets is not None and v in self.targets else html.escape(v.name)}=' + (f'{html.escape(str(dist.expectation()))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items()])}</TD>
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

            dot.node(str(idx),
                     label=lbl,
                     shape='box',
                     style='rounded,filled',
                     fillcolor=green)
        for idx, node in self.innernodes.items():
            dot.node(str(idx),
                     label=node.str_node,
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

    def pickle(self, fpath: str) -> None:
        """
        Pickles the fitted regression tree to a file at the given location ``fpath``.

        :param fpath: the location for the pickled file
        """
        with open(os.path.abspath(fpath), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fpath) -> 'JPT':
        """
        Loads the pickled regression tree from the file at the given location ``fpath``.

        :param fpath: the location of the pickled file
        :type fpath: str
        """
        with open(os.path.abspath(fpath), 'rb') as f:
            try:
                JPT.logger.info(f'Loading JPT {os.path.abspath(fpath)}')
                return pickle.load(f)
            except ModuleNotFoundError:
                JPT.logger.error(
                    f'Could not load file {os.path.abspath(fpath)}'
                )
                raise Exception(
                    f'Could not load file {os.path.abspath(fpath)}. Probably deprecated.'
                )

    @staticmethod
    def calcnorm(sigma: float, mu: float, intervals):
        """
        Computes the CDF for a multivariate normal distribution.

        :param sigma: the standard deviation
        :param mu: the expected value
        :param intervals: the boundaries of the integral
        :type intervals: list of matcalo.utils.utils.Interval
        :return:
        """
        from scipy.stats import mvn
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], mu, sigma))

    def copy(self) -> 'JPT':
        """
        :return: a new copy of this jpt where all references are the original tree are cut.
        """
        return JPT.from_json(self.to_json())

    def conditional_jpt(
            self,
            evidence: Optional[VariableAssignment] = None,
            fail_on_unsatisfiability: bool = True
    ) -> 'JPT' or None:
        """
        Apply evidence on a JPT and get a new JPT that represent P(x|evidence).

        :param evidence: A VariableAssignment mapping the observed variables to there observed values
         :param fail_on_unsatisfiability: whether an error is raised in case of unsatisfiable evidence or not
        """

        if not evidence:
            evidence = LabelAssignment()

        # Convert, if necessary, labels to internal value representations
        if isinstance(evidence, LabelAssignment):
            evidence = evidence.value_assignment()

        # the new jpt that acts as conditional joint probability distribution
        conditional_jpt: JPT = self.copy()

        # skip if evidence is empty
        if not evidence:
            return conditional_jpt

        # initialize exploration queue
        fringe = deque([conditional_jpt.root])

        while fringe:
            # get the next node to inspect
            node = fringe.popleft()

            # the node might have been deleted already
            if node not in conditional_jpt.allnodes.values():
                continue

            # initialize remove as false
            rm = False

            # if it is a decision node
            if isinstance(node, DecisionNode):

                # that has no children
                if len(node.children) <= 1:

                    # remove it and update flag
                    del conditional_jpt.innernodes[node.idx]
                    rm = True

                # else recurse into its children
                else:
                    fringe.extendleft(node.children)
                    continue

            # if it is a leaf node
            else:
                # type hinting
                leaf: Leaf = node

                # calculate probability of this leaf being selected given the evidence
                probability = leaf.probability(evidence)

                # if the leafs probability is 0 with the evidence
                if probability > 0:
                    leaf.prior *= probability
                else:
                    # remove the leaf and set the flag
                    rm = True
                    del conditional_jpt.leaves[node.idx]

            # if the node has been removed and the removed node as a parent
            if rm and node.parent is not None:

                # if the nodes parent has children
                if node.parent.children:

                    # and the node has a child either
                    if isinstance(node, DecisionNode) and len(node.children) == 1:

                        # replace it by its child
                        node.parent.children[node.parent.children.index(node)] = first(node.children)

                    # delete this node from the parents children
                    else:
                        idx = node.parent.children.index(node)
                        del node.parent.children[idx]
                        del node.parent.splits[idx]

                # append the parent node the queue
                fringe.append(node.parent)

            # if the resulting model is empty
            if rm and node is conditional_jpt.root:

                if len(node.children) == 1:
                    conditional_jpt.root = first(node.children)

                # raise an error if wanted
                elif fail_on_unsatisfiability:
                    raise Unsatisfiability(
                        'Query is unsatisfiable: P(%s) is 0.' % format_path(evidence)
                    )

                # return None if error is not wanted
                else:
                    return None

        # calculate remaining probability mass
        probability_mass = sum(leaf.prior for leaf in conditional_jpt.leaves.values())

        if not probability_mass:
            raise Unsatisfiability(
                'JPT is unsatisfiable (all %d leaves have 0 prior probability)' % len(self)
            )

        # clean up not needed distributions and redistribute probability mass
        for leaf in conditional_jpt.leaves.values():

            # normalize probability
            leaf.prior /= probability_mass

            for variable, value in evidence.items():
                # adjust leaf distributions
                if variable.symbolic or variable.integer:
                    leaf.distributions[variable] = leaf.distributions[variable]._crop(value)

                # for numeric variables it's not as straight forward due to the value being polymorph
                elif variable.numeric:
                    leaf.distributions[variable] = leaf.distributions[variable].crop(value)

                else:
                    raise ValueError("Unknown variable type to crop. Type is %s" % type(variable))

        # clean up not needed path restrictions
        for node in conditional_jpt.allnodes.values():
            for variable in evidence.keys():
                if variable in node.path.keys():
                    del node.path[variable]

        # recalculate the priors for the conditional jpt
        priors = conditional_jpt.posterior(
            evidence=conditional_jpt.bind({v.name: e for v, e in evidence.label_assignment().items()}))
        conditional_jpt.priors = priors

        return conditional_jpt

    def multiply_by_leaf_prior(self, prior: Dict[int, float]) -> 'JPT':
        """
        Multiply every leafs prior by the given priors. This serves as handling the factor message
        from factor nodes. Be vary since this method overwrites the JPT in-place.

        :param prior: The priors, a Dict mapping from leaf indices to float

        :return: self
        """
        for idx, leaf in self.leaves.items():
            self.leaves[idx].prior *= prior[idx]
        self.normalize()
        return self

    def normalize(self) -> 'JPT':
        """
        Normalize the tree s. t. the sum of all leaf priors is 1.
        :return: self
        """
        probability_mass = sum(leaf.prior for leaf in self.leaves.values())
        if not probability_mass:
            raise Unsatisfiability(
                'JPT is unsatisfiable (all %s leaves have 0 prior probability).' % len(self.leaves)
            )
        for idx, leaf in self.leaves.items():
            self.leaves[idx].prior /= probability_mass
        return self

    def save(self, file) -> None:
        """
        Write this JPT persistently to disk.
        :param file: either a string or file-like object.
        """
        if type(file) is str:
            with open(file, 'w+') as f:
                json.dump(self.to_json(), f)
        else:
            json.dump(self.to_json(), file)

    @staticmethod
    def load(file) -> 'JPT':
        """
        Load a JPT from disk.
        :param file: either a string or file-like object.
        :return: the JPT described in ``file``
        """
        if type(file) is str:
            with open(file, 'r') as f:
                t = json.load(f)
        else:
            t = json.load(file)
        return JPT.from_json(t)

    def depth(self) -> int:
        """
        :return: the maximal depth of a leaf in the tree.
        """
        return max([leaf.depth() for leaf in self.leaves.values()])

    def total_samples(self) -> int:
        """
        :return: the total number of samples represented by this tree.
        """
        return sum(l.samples for l in self.leaves.values())

    def postprocess_leaves(self) -> None:
        """Postprocess leaves such that the convex hull that is postulated from this tree has likelihood > 0 for every
        point inside the hull."""

        # get total number of samples and use 1/total as default value
        total_samples = self.total_samples()

        # for every leaf
        for idx, leaf in self.leaves.items():
            # for numeric every distribution
            for variable, distribution in leaf.distributions.items():
                if variable.numeric and variable in leaf.path.keys() and not distribution.is_dirac_impulse():

                    left = None
                    right = None

                    # if the leaf is not the "lowest" in this dimension
                    if -float("inf") < leaf.path[variable].lower < distribution.cdf.intervals[0].upper:
                        # create uniform distribution as bridge between the leaves
                        interval = ContinuousSet(
                            leaf.path[variable].lower,
                            distribution.cdf.intervals[0].upper
                        )
                        left = interval

                    # if the leaf is not the "highest" in this dimension
                    if float("inf") > leaf.path[variable].upper > distribution.cdf.intervals[-2].upper:
                        # create uniform distribution as bridge between the leaves
                        interval = ContinuousSet(
                            distribution.cdf.intervals[-2].upper,
                            leaf.path[variable].upper
                        )
                        right = interval

                    distribution.insert_convex_fragments(left, right, total_samples)

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in the entire tree
        """
        return sum([node.number_of_parameters() for node in self.leaves.values()])

    # noinspection PyIncorrectDocstring
    def bind(self, *arg, **kwargs) -> LabelAssignment:
        '''
        Returns a ``LabelAssignment`` object with the assignments passed.

        This method accepts one optional positional argument, which -- if passed -- must be a dictionary
        of the desired variable assignments.

        Keyword arguments may specify additional variable, value pairs.

        If a positional argument is passed, the following options may be passed in addition
        as keyword arguments:

        :param allow_singular_values: Allow singular values, such that they are transformed to the daomain
            specification of numeric variables but not transformed to intervals via the PPF.
        '''
        options = {'allow_singular_values': False}
        if len(arg) > 1 or arg and not isinstance(arg[0], dict) and not arg[0] is None:
            raise ValueError(
                'Illegal argument: positional '
                'argument of bind() must be a dict, got %s' % type(arg[0]).__name__
            )
        elif len(arg):
            if set(kwargs).difference(options):
                raise ValueError(
                    'Options of bind() must be a subset of %s, got %s.' % (set(options), set(kwargs))
                )
            bindings = ifnone(arg[0], {})
            options.update(kwargs)
        else:
            bindings = kwargs
        return self._preprocess_query(bindings, **options)

    def sample(self, amount) -> np.array:
        """Sample `amount` many samples from the tree.

        :returns A numpy array of size (amount, self.variables) containing the samples.
        """
        # create probability distribution for the leaves
        leaf_probabilities = np.array([leaf.prior for leaf in self.leaves.values()])

        # sample in which function part the samples will be
        sampled_leaves = np.random.choice(list(self.leaves.keys()), size=(amount,), p=leaf_probabilities)

        samples = np.empty((amount, len(self.variables)), dtype=object)

        # for every leaf
        for idx, leaf in self.leaves.items():

            # get indices of samples in this leaf
            indices = (sampled_leaves == idx).nonzero()[0]

            # skip if empty
            if len(indices) == 0:
                continue

            leaf_samples = leaf.sample(len(indices))
            samples[indices] = leaf_samples

        return samples

    def moment(self, order: int = 1, center: Optional[VariableAssignment] = None,
               evidence: Optional[VariableAssignment] = None,
               fail_on_unsatisfiability: bool = True, ) -> VariableMap or None:
        """ Calculate the order of each numeric/integer random variable given the evidence.

        :param order: The order of the moment
        :param center: A VariableAssignment mapping each numeric/integer variable to some constant.
            If a variable has a constant, it will be interpreted as 'c' for the central moment.
            If it is not set, 0 will be used by default.
        :param evidence: The evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: Rather or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        """

        if not center:
            center = LabelAssignment()

        if not evidence:
            evidence = LabelAssignment()

        # calculate posterior distributions
        posteriors = self.posterior([v for v in self.variables if v.numeric or v.integer], evidence,
                                    fail_on_unsatisfiability)

        # Convert c, if necessary, labels to internal value representations
        if isinstance(center, LabelAssignment):
            center = center.value_assignment()

        if posteriors is None:
            return None

        result = dict()

        for variable, distribution in posteriors.items():
            if variable not in center:
                current_c = 0
            else:
                current_c = center[variable]

            result[variable] = distribution.moment(order, current_c)
        return VariableMap(result.items())

    def get_hyperparameters_dict(self) -> Dict[str, Any]:
        """Get all hyperparameters as dict that can be used for MLFlow model tracking."""
        hyperparameters = dict()
        hyperparameters["variables"] = [v.name for v in self.variables]

        for variable in self.variables:
            json_dict = variable.to_json()

            for setting, value in json_dict["settings"].items():
                hyperparameters[f"{variable.name}.{setting}"] = value

        hyperparameters["targets"] = [v.name for v in self.targets]
        hyperparameters["features"] = [v.name for v in self.features]
        hyperparameters["min_samples_leaf"] = self.min_samples_leaf
        hyperparameters["min_impurity_improvement"] = self.min_impurity_improvement
        hyperparameters["max_leaves"] = self.max_leaves
        hyperparameters["max_depth"] = self.max_depth

        return hyperparameters
