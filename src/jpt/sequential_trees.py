from builtins import dict
from typing import List

import numpy as np
import numpy.lib.stride_tricks
import pandas
import pandas as pd

from fglib import graphs, nodes, rv, inference
import factorgraph
import jpt.trees
from jpt.base.errors import Unsatisfiability
from jpt.base.utils import format_path


class SequentialJPT:
    def __init__(self, template_tree):
        self.template_tree: jpt.trees.JPT = template_tree
        self.transition_model: np.array or None = None

    def fit(self, sequences: List[np.ndarray or pd.Series or pd.DataFrame], timesteps: int = 2):
        """ Fits the transition and emission models. The emission model is fitted
         with respect to the variables in the next timestep, but it doesn't use them.

         @param sequences: The sequences to learn from
         @param timesteps: The timesteps to jointly model (minimum of 2 required) """

        # extract copies of variables for the expanded tree
        expanded_variables = [var.copy() for var in self.template_tree.variables]

        # keep track of which dimensions to include in the training process
        data_indices = list(range(len(expanded_variables)))

        # extract target indices from the
        if self.template_tree.targets:
            target_indices = [idx for idx, var in enumerate(self.template_tree.variables)
                              if var in self.template_tree.targets]
        else:
            target_indices = list(range(len(self.template_tree.variables)))

        # create variables for jointly modelled timesteps
        for timestep in range(1, timesteps):
            expanded_variables += [self._shift_variable_to_timestep(self.template_tree.variables[idx], timestep) for idx
                                   in target_indices]

            # append targets to data index
            data_indices += [idx + timestep * len(self.template_tree.variables) for idx in target_indices]

        # create expanded tree
        expanded_template_tree = jpt.trees.JPT(variables=expanded_variables,
                                               targets=expanded_variables[len(self.template_tree.variables):],
                                               min_samples_leaf=self.template_tree.min_samples_leaf,
                                               min_impurity_improvement=self.template_tree.min_impurity_improvement,
                                               max_leaves=self.template_tree.max_leaves,
                                               max_depth=self.template_tree.max_depth)

        # initialize data
        data = None

        # convert pandas types to numpy
        for idx, sequence in enumerate(sequences):
            if isinstance(sequence, pd.Series) or isinstance(sequence, pd.DataFrame):
                sequences[idx] = sequence.to_numpy()
            if len(sequences[idx].shape) == 1:
                sequences[idx] = sequences[idx].reshape(-1, 1)

        # for every sequence
        for sequence in sequences:

            # unfold the timesteps such that they are expanded to jointly model all timesteps
            unfolded = np.lib.stride_tricks.sliding_window_view(sequence, (timesteps, ), axis=0)
            unfolded = unfolded.reshape((len(unfolded), len(self.template_tree.variables) * timesteps), order="F")

            unfolded = unfolded[:, data_indices]

            # append or set data
            if data is None:
                data = unfolded
            else:
                data = np.concatenate((data, unfolded), axis=0)

        # fit joint timesteps tree
        expanded_template_tree.learn(data=data)

        # create template tree from learnt joint tree
        self.template_tree.root = expanded_template_tree.root
        self.template_tree.innernodes = expanded_template_tree.innernodes
        for idx, leaf in expanded_template_tree.leaves.items():
            leaf.distributions = jpt.variables.VariableMap([(v, d) for v, d in leaf.distributions.items()
                                                            if v.name in self.template_tree.varnames.keys()])
            self.template_tree.leaves[idx] = leaf

        transition_data = None

        for sequence in sequences:

            # encode the samples to 'leaf space'
            encoded = self.template_tree.encode(sequence)

            # convert to 2 sizes sliding window
            transitions = numpy.lib.stride_tricks.sliding_window_view(encoded, (2,), axis=0)

            # concatenate transitions
            if transition_data is None:
                transition_data = transitions
            else:
                transition_data = np.concatenate((transition_data, transitions))

        # load number of leaves
        num_leaves = len(self.template_tree.leaves)

        # calculate factor values for transition model
        values = np.zeros((num_leaves, num_leaves))
        for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
            for jdx, leaf_jdx in enumerate(self.template_tree.leaves.keys()):
                count = sum((transition_data[:, 0] == leaf_idx) & (transition_data[:, 1] == leaf_jdx))
                values[idx, jdx] = count/len(transition_data)

        self.transition_model = values

    def _shift_variable_to_timestep(self, variable: jpt.variables.Variable, timestep: int = 1) -> jpt.variables.Variable:
        """ Create a new variable where the name is shifted by +n and the domain remains the same.

        @param variable: The variable to shift
        @param timestep: timestep in the future, i.e. timestep >= 1
        """
        variable_ = variable.copy()
        variable_._name = "%s+%s" % (variable_.name, timestep)
        return variable_

    def preprocess_sequence_map(self, evidence: List[jpt.variables.VariableMap]):
        """ Preprocess a list of variable maps to be used in JPTs. """
        return [self.template_tree._preprocess_query(e) for e in evidence]

    def ground(self, evidence: List[jpt.variables.VariableMap]) -> (factorgraph.Graph, List[jpt.trees.JPT]):
        """Ground a factor graph where inference can be done. The factor graph is grounded with
        one variable for each timestep, one prior node as factor for each timestep and one factor node for each
        transition.

        @param evidence: A list of VariableMaps that describe evidence in the given timesteps.
        """

        # create factorgraph
        factor_graph = factorgraph.Graph()

        # add variable nodes for timesteps
        timesteps = ["t%s" % t for t in range(len(evidence))]
        [factor_graph.rv(timestep, len(self.template_tree.leaves)) for timestep in timesteps]

        altered_jpts = []

        # for each transition
        for idx in range(len(evidence)-1):

            # get the variable names
            state_names = ["t%s" % idx, "t%s" % (idx+1)]

            # create factor with values from transition model
            factor_graph.factor(state_names, potential=self.transition_model)

        # create prior factors
        for timestep, e in zip(timesteps, evidence):

            # apply the evidence
            conditional_jpt = self.template_tree.conditional_jpt(e)

            # append altered jpt
            altered_jpts.append(conditional_jpt)

            # create the prior distribution from the conditional tree
            prior = np.zeros((len(self.template_tree.leaves), ))

            # fill the distribution with the correct values
            for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
                if leaf_idx in conditional_jpt.leaves.keys():
                    prior[idx] = conditional_jpt.leaves[leaf_idx].prior

            # create a factor from it
            factor_graph.factor([timestep], potential=prior)

        return factor_graph, altered_jpts

    def ground_fglib(self, evidence: List[jpt.variables.VariableMap]) -> (graphs.FactorGraph, List[jpt.trees.JPT]):
        """Ground a factor graph where inference can be done. The factor graph is grounded with
        one variable for each timestep, one prior node as factor for each timestep and one factor node for each
        transition.

        @param evidence: A list of VariableMaps that describe evidence in the given timesteps.
        """

        # create factorgraph
        factor_graph = graphs.FactorGraph()

        # add variable nodes for timesteps
        timesteps = ["t%s" % t for t in range(len(evidence))]
        fg_variables = [nodes.VNode(t, rv.Discrete) for t in timesteps]
        factor_graph.set_nodes(fg_variables)

        fg_factors = []

        # for each transition
        for idx in range(len(evidence)-1):

            # create factor with values from transition model
            current_factor = nodes.FNode("P(%s,%s)" % ("t%s" % idx, "t%s" % (idx+1)),
                                         rv.Discrete(self.transition_model, fg_variables[idx], fg_variables[idx+1]))
            factor_graph.set_node(current_factor)
            factor_graph.set_edge(fg_variables[idx], current_factor)
            factor_graph.set_edge(fg_variables[idx+1], current_factor)

        altered_jpts = []

        # create prior factors
        for fg_variable, e in zip(fg_variables, evidence):

            # apply the evidence
            conditional_jpt = self.template_tree.conditional_jpt(e)

            # append altered jpt
            altered_jpts.append(conditional_jpt)

            # create the prior distribution from the conditional tree
            prior = np.zeros((len(self.template_tree.leaves), ))

            # fill the distribution with the correct values
            for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
                if leaf_idx in conditional_jpt.leaves.keys():
                    prior[idx] = conditional_jpt.leaves[leaf_idx].prior

            # create a factor from it
            current_factor = nodes.FNode("P(%s)" % str(fg_variable), rv.Discrete(prior, fg_variable))
            factor_graph.set_node(current_factor)
            factor_graph.set_edge(fg_variable, current_factor)

        return factor_graph, altered_jpts

    def mpe(self, evidence):
        raise NotImplementedError("Not yet implemented")

    def leaf_distribution(self, query) -> np.array:
        result = np.zeros(len(self.template_tree.leaves.values()))
        for idx, leaf in enumerate(self.template_tree.leaves.values()):
            result[idx] = leaf.probability(self.template_tree.bind(query)) * leaf.prior
        # Hier kann es dazu kommen, dass sum(result) == 0 ist
        if sum(result) == 0:
            return np.array([0, 0])
        return result/sum(result)

    def infer(self, query, evidence) -> float:
        """
        Calculate the probability of sequence 'query' given sequence 'evidence'.

        @param query: The question
        @param evidence: The evidence
        @return: probability (float)
        """

        idx_leaves_list = {}
        for idx, key in enumerate(self.template_tree.leaves.keys()):
            idx_leaves_list[key] = idx

        # Calculating P(E)

        # Calculating leaf transition probability
        # P(Lam_t+1 | lam_t) = Sum_t(P(Lam_t+1 | lam_t) * lam_t)
        p_leaf_t = np.ones(len(evidence), dtype=object)

        for idx, evidence_t in enumerate(evidence[1:]):
            if not evidence_t:
                p_leaf_t[idx+1] = np.sum(self.transition_model, axis=1)
                continue
            pre_leaf_encoding = self.leaf_distribution(evidence[idx])
            current_leaf_encoding = self.leaf_distribution(evidence_t)
            if idx == 0:
                p_leaf_t[idx] = pre_leaf_encoding
            result_t_iteration = 0
            # Pred.
            for pre_leaf_distribution in p_leaf_t[idx]:
                result_t_iteration = current_leaf_encoding * pre_leaf_distribution
            # Upd.
            # Hier kann es dazu kommen, dass sum(current_leaf_encoding * result_t_iteration)) == 0 ist
            if sum(current_leaf_encoding * result_t_iteration) == 0:
                p_leaf_t[idx + 1] = np.array([0,0])
            else:
                trans_prob = (current_leaf_encoding * result_t_iteration) / sum(current_leaf_encoding * result_t_iteration)
                p_leaf_t[idx+1] = trans_prob
        print("Leaf trainsition: ", p_leaf_t)

        # Calculating P(Ex | leave)
        p_e_y = 1
        p_evidence_given_leave = np.zeros(len(evidence), dtype=object)
        for idx, evidence_t in enumerate(evidence):
            if evidence_t:
                query_leaf_parallel_likelihood = np.zeros(len(self.template_tree.leaves.values()), dtype=object)
                for idx_leaf, leaf in enumerate(self.template_tree.leaves.values()):
                    p_e_query = [[x] for x in list(*evidence_t.values())]
                    query_leaf_parallel_likelihood[idx_leaf] = leaf.probability(queries=np.array(p_e_query),
                                                                                        min_distances=self.template_tree.minimal_distances)
                p_evidence_given_leave[idx] = query_leaf_parallel_likelihood
        print("P(Ex | leave): ", p_evidence_given_leave)


        ####
        # Calculating P(Q,E)
        p_q_e_result_list = np.zeros(len(query), dtype=object)
        p_q_e_list = np.zeros(len(query), dtype=object)
        for idx, sequence in enumerate(query):
            # Calculation intersection
            if evidence[idx]:
                sequence[list(sequence.keys())[0]] = [max(list(evidence[idx].values())[0][0],
                                                          list(sequence.values())[0][0]),
                                                      min(list(evidence[idx].values())[0][1],
                                                          list(sequence.values())[0][1])]

            encode_seq = self.template_tree.encode(np.array([[x] for x in list(*sequence.values())]))
            p_q_e_list[idx] = [p_leaf_t[idx][int(x)-1] for x in list(encode_seq)]
            print(self.template_tree.leaves[2].parallel_likelihood(queries=np.array([[x] for x in list(*sequence.values())]),
                                                                   min_distances=self.template_tree.minimal_distances))
            p_q_e_result_list[idx] = [p_leaf_t[idx][int(x)-1] * self.template_tree.leaves[x].parallel_likelihood(queries=np.array([[x] for x in list(*sequence.values())]),
                                                                                                             min_distances=self.template_tree.minimal_distances) for x in list(encode_seq)]
        print("P(Q,E): ", p_q_e_result_list)

        # Calculate P(E)
        p_e = 1
        for idx, evi in enumerate(p_evidence_given_leave):
            p_e *= p_leaf_t[idx] * evi

        # Calculating P(Q,E)
        p_q_e = 1
        for q_e in p_q_e_result_list:
            p_q_e *= np.array(q_e)


        print("P(E): ", p_e)
        print("P(Q,E): ", p_q_e)
        if sum(sum(p_e)) != 0:
            return p_q_e / p_e
        else:
            raise Unsatisfiability('Evidence %s is unsatisfiable.' % evidence)

    def posterior(self, evidence: List[jpt.variables.VariableMap]) -> List[jpt.trees.JPT]:
        """
        :param evidence:
        :return:
        """
        from fglib import utils
        # preprocess evidence
        evidence = self.preprocess_sequence_map(evidence)

        # ground factor graph
        factor_graph, altered_jpts = self.ground_fglib(evidence)

        # create result list
        result = []

        # Run belief propagation
        latent_distribution = []
        for v_node in factor_graph.get_vnodes():
            belief = inference.belief_propagation(graph=factor_graph, query_node=v_node)
            latent_distribution.append(belief)

        # transform trees
        for distribution, tree in zip(latent_distribution, altered_jpts):
            prior = dict(zip(self.template_tree.leaves.keys(), distribution.pmf))
            adjusted_tree = tree.multiply_by_leaf_prior(prior)
            result.append(adjusted_tree)

        return result

    def independent_marginals(self, evidence: List[jpt.variables.VariableMap]) -> List[jpt.trees.JPT]:
        """ Return the independent marginal distributions of all variables in this sequence along all
        timesteps.

        @param evidence: The evidence observed in every timesteps. The length of this list determines the length
            of the whole sequence
        """
        # preprocess evidence
        evidence = self.preprocess_sequence_map(evidence)

        # ground factor graph
        factor_graph, altered_jpts = self.ground(evidence)

        # create result list
        result = []

        # Run (loopy) belief propagation (LBP)
        iters, converged = factor_graph.lbp(max_iters=100, progress=True)
        latent_distribution = factor_graph.rv_marginals()

        # transform trees
        for ((name, distribution), tree) in zip(sorted(latent_distribution, key=lambda x: x[0].name), altered_jpts):
            prior = dict(zip(self.template_tree.leaves.keys(), distribution))
            adjusted_tree = tree.multiply_by_leaf_prior(prior)
            result.append(adjusted_tree)

        return result

    def expectation(self,
                    variables: List[List[jpt.variables.Variable]],
                    evidence: List[jpt.variables.VariableAssignment],
                    confidence_level: float = None,
                    fail_on_unsatisfiability: bool = True) -> List[jpt.variables.VariableMap]:
        """
        :param variables:
        :param evidence:
        :param confidence_level:
        :param fail_on_unsatisfiability:
        :return:
        """
        posteriors = self.posterior(evidence=evidence)

        final = []
        for idx, tree in enumerate(posteriors):
            final.append(tree.expectation(variables=variables[idx],
                                          evidence=evidence[idx],
                                          fail_on_unsatisfiability=fail_on_unsatisfiability))

        return final

    def likelihood(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for sequence in sequences:
            leaf_likelihood = np.zeros(sequence.size)
            prior_index_leaf = None
            for leaf in self.template_tree.leaves.values():
                if prior_index_leaf is None:
                    prior_index_leaf = leaf.prior
                print(sequence)
                leaf_likelihood = np.add(leaf_likelihood, leaf.parallel_likelihood(queries=sequence,
                                                                                   min_distances=self.template_tree.minimal_distances))

            idx_leaves_list = {}
            for idx, key in enumerate(self.template_tree.leaves.keys()):
                idx_leaves_list[key] = idx

            leaf_transition_probability = []
            print(sequence)
            for pre_leaf_idx, leaf_idx in zip(self.template_tree.encode(sequence), self.template_tree.encode(sequence)[1:]):
                leaf_transition_probability.append(self.transition_model[idx_leaves_list.get(leaf_idx)][idx_leaves_list.get(pre_leaf_idx)] / self.template_tree.leaves.get(leaf_idx).prior)

            print(self.transition_model[idx_leaves_list.get(leaf_idx)][idx_leaves_list.get(pre_leaf_idx)])
            print(self.template_tree.leaves.get(leaf_idx).prior)

            leaf_transition_probability = np.array(leaf_transition_probability)

            sequence_likelihood = []
            for idx in range(len(sequence)):
                if idx is 0:
                    sequence_likelihood.append(prior_index_leaf) # P(y0)
                    sequence_likelihood.append(leaf_likelihood[idx]) # P(yi | Xi)
                else:
                    sequence_likelihood.append(leaf_transition_probability[idx-1]) # P(yi | yj) | i > j
                    sequence_likelihood.append(leaf_likelihood[idx]) # P(yi | Xi)

            sequence_likelihood = np.array(sequence_likelihood)
            result.append(sequence_likelihood)

        result = np.array(result, dtype=object)
        return result

    def to_json(self):
        return {"template_tree": self.template_tree.to_json(),
                "transition_model": self.transition_model.tolist()}

    @staticmethod
    def from_json(data):
        template_tree = jpt.trees.JPT.from_json(data["template_tree"])
        result = SequentialJPT(template_tree)
        result.transition_model = np.array(data["transition_model"])
        return result