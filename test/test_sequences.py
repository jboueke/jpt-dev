import unittest
import numpy as np
import pandas as pd
import jpt.variables
import jpt.trees
from jpt.distributions.univariate import SymbolicType
import jpt.sequential_trees


class UniformSeries:

    def __init__(self, basis_function=np.sin, epsilon=0.05):
        self.epsilon = 0.05
        self.basis_function = basis_function

    def sample(self, samples) -> np.array:
        samples = self.basis_function(samples)
        samples = samples + np.random.uniform(-self.epsilon, self.epsilon, samples.shape)
        return samples


class SequenceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.g = UniformSeries()
        self.data = np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1).reshape(-1, 1)
        self.variables = [jpt.variables.NumericVariable("X", precision=0.1)]

    def test_learning(self):
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree.fit([self.data, self.data])

        r = sequence_tree.independent_marginals([
            {},
            template_tree.bind(X=[0.95, 1.05]),
            {}
        ])

        for tree in r:
            self.assertEqual(sum(l.prior for l in tree.leaves.values()), 1.)


class SequenceTestFglib(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(420)
        self.g = UniformSeries()
        self.data = np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1).reshape(-1, 1)
        self.data_one_dimensional = self.g.sample(np.arange(np.pi / 2, 10000, np.pi))
        self.pd_series_data = pd.Series(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)))
        self.pd_dataframe_data = pd.DataFrame(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)))
        self.variables = [jpt.variables.NumericVariable("X", precision=0.1)]

    def test_learning(self):
        sequence_trees = []
        template_tree_np = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        template_tree_np_one_dimensional = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        template_tree_pd_series = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        template_tree_pd_dataframe = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree_np = jpt.sequential_trees.SequentialJPT(template_tree_np)
        sequence_tree_np_one_dimensional = jpt.sequential_trees.SequentialJPT(template_tree_np_one_dimensional)
        sequence_tree_pd_series = jpt.sequential_trees.SequentialJPT(template_tree_pd_series)
        sequence_tree_pd_dataframe = jpt.sequential_trees.SequentialJPT(template_tree_pd_dataframe)
        sequence_tree_np.fit([self.data, self.data])
        sequence_trees.append(sequence_tree_np)
        sequence_tree_np_one_dimensional.fit([self.data_one_dimensional, self.data_one_dimensional])
        sequence_trees.append(sequence_tree_np_one_dimensional)
        sequence_tree_pd_series.fit([self.pd_series_data, self.pd_series_data])
        sequence_trees.append(sequence_tree_pd_series)
        sequence_tree_pd_dataframe.fit([self.pd_dataframe_data, self.pd_dataframe_data])
        sequence_trees.append(sequence_tree_pd_dataframe)

        evidences = [{}, jpt.variables.VariableMap({self.variables[0]: [0.95, 1.05]}.items()), {}]

        for sequence_tree in sequence_trees:
            result = sequence_tree.posterior(evidences)

            for idx, tree in enumerate(result):
                expectation = tree.expectation(["X"])
                if idx % 2 == 0:
                    self.assertAlmostEqual(expectation["X"], -1., delta=0.001)
                else:
                    self.assertAlmostEqual(expectation["X"], 1., delta=0.001)

    def test_expectation(self):
        sequential_variables = [[jpt.variables.NumericVariable("X", precision=0.1)],
                                [jpt.variables.NumericVariable("X", precision=0.1)],
                                [jpt.variables.NumericVariable("X", precision=0.1)]]
        template_tree = jpt.trees.JPT(sequential_variables[0], min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree.fit([self.data, self.data])

        evidences = [{}, {"X": [0.95, 1.05]}, {}]

        result = sequence_tree.expectation(variables=sequential_variables, evidence=evidences)

        for idx, expectation_result in enumerate(result):
            expectation = expectation_result["X"]
            if idx % 2 == 0:
                self.assertAlmostEqual(expectation, -1, delta=0.001)
            else:
                self.assertAlmostEqual(expectation, 1, delta=0.001)

    def test_likelihood(self):
        sequential_variables = [[jpt.variables.NumericVariable("X", precision=0.1)],
                                [jpt.variables.NumericVariable("X", precision=0.1)],
                                [jpt.variables.NumericVariable("X", precision=0.1)]]
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree.fit([self.data, self.data, self.data[:20]])

        result = sequence_tree.likelihood([self.data, self.data, self.data[:20]])
        # sequence_tree.template_tree.plot(plotvars=sequence_tree.template_tree.variables)
        self.assertTrue(all(all(sequence > 0) for sequence in result))

    def test_different_likelohood(self):
        template_tree_less_leaves = jpt.trees.JPT(self.variables, min_samples_leaf=0.9)
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree_less_leaves = jpt.sequential_trees.SequentialJPT(template_tree_less_leaves)
        sequence_tree.fit([self.data, self.data, self.data[:20]])
        sequence_tree_less_leaves.fit([self.data, self.data, self.data[:20]])

        result_less_leaves = sequence_tree_less_leaves.likelihood([self.data, self.data, self.data[:20]])
        result = sequence_tree.likelihood([self.data, self.data, self.data[:20]])

        # sequence_tree_less_leaves.template_tree.plot(plotvars=sequence_tree_less_leaves.template_tree.variables)
        self.assertTrue(sum(sum(np.log(sequence)) for sequence in result) >
                        sum(sum(np.log(sequence)) for sequence in result_less_leaves))

    def test_infer(self):
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree.fit([self.data, self.data])

        query = [{"X": [0.9, 1]}, {"X": [-0.9, -0.89]}, {"X": [0.95, 1.05]}]
        evidences = [{"X": [0.9, 1.0]}, {"X": [-1.05, -0.9]}, {}]

        query2 = [{"X": [0.5, 1]}, {"X": [-1.9, -1.02]}, {"X": [0.95, 1.05]}]
        evidences2 = [{"X": [0.9, 1.0]}, {"X": [-1.95, -1.05]}, {}]

        result = sequence_tree.infer(query, evidences)
        print(result)

if __name__ == '__main__':
    unittest.main()