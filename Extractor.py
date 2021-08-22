import math
import numpy as np
from silas import add_f, sub_f


class Extractor(object):
    def __init__(self, estimators=None, phi=0, theta=0, psi=0):
        """Build a rule extractor.

        :param estimators: random forest
        :param phi: rule filter threshold
        :param theta: node filter threshold
        :param psi: leaf merger to create class signature
        """
        self._estimators = estimators
        self._forest_paths = []         # all forest paths
        self.scale = 0
        self._forest_formulae = []      # all paths after filter, each element with shape (2, n_feature)
                                        # representing min and max value
        self._forest_formulae_visited = []  # all visited rules
        self._forest_values = []        # all node votes after filter
        self._forest_weights = []       # all rule weights after filter
        self._forest_signatures = []    # all rule signatures after filter
        self.n_estimators = estimators.n_estimators
        self.n_classes = estimators.n_classes_
        self.n_features = estimators.n_features_
        self.n_outputs = estimators.n_outputs_
        self._phi = phi
        self._theta = theta
        self._psi = psi
        self.n_original_leaves_num = 0

        self._quality = []
        self._ig = []
        self.max_rule = 0   # record the max rule quality
        self.max_node = 0   # record the max node information gain
        self.min_rule = 1
        self.min_node = 1

    def set_param(self, phi, theta, psi):
        """Set all three parameters."""
        self._phi = phi
        self._theta = theta
        self._psi = psi

    def opt_set_quality(self, quality, ig):
        self._quality = quality
        self._ig = ig

    def opt_get_quality(self):
        return self._quality, self._ig

    def opt_clear_quality(self):
        self._quality = []
        self._ig = []

    def extract_tree_paths(self, estimator):
        return estimator.rules

    def extract_forest_paths(self):
        for estimator in self._estimators:
            paths = self.extract_tree_paths(estimator)
            for path in paths:
                self.scale += (len(path) - 1)
            self._forest_paths.append(paths)

    def count_quality(self):
        _n_classes = self._estimators.n_classes_
        for index, tree in enumerate(self._estimators.trees_):
            acc = self._estimators.trees_oob_scores[index]
            _tree = self._estimators[index]
            rule_quality = []
            node_quality = []
            for rule in _tree:
                _node_quality = []
                self.n_original_leaves_num += 1

                quality = (1 - (_tree.ig[rule[-1]] / math.log(_n_classes, 2))) * acc  # rule quality, Eq (7)
                rule_quality.append(quality)

                if quality > self.max_rule:
                    self.max_rule = quality
                if quality < self.min_rule:
                    self.min_rule = quality

                for node in rule[:-1]:
                    ig = _tree.ig[node]
                    _node_quality.append(ig)
                    if ig > self.max_node:
                        self.max_node = ig
                    if ig < self.min_node:
                        self.min_node = ig
                node_quality.append(_node_quality)
            self._quality.append(rule_quality)
            self._ig.append(node_quality)

    def rule_filter(self):
        if len(self._quality) == 0:
            self.count_quality()

        if self._phi > self.max_rule or self._theta > self.max_node:
            return False

        length = len(self._forest_signatures)
        for index, tree in enumerate(self._estimators.trees_):
            for j, rule in enumerate(tree):
                quality = self._quality[index][j]
                if quality >= self._phi:  # rule filter: rq >= phi
                    if self.node_filter(j, index) == 1:
                        signature = np.array(self._estimators[index].value[rule[-1]])
                        leaf_sum = self._estimators[index].value[rule[-1]].sum()
                        signature = np.ceil(signature / leaf_sum / self._psi)
                        self._forest_signatures.append(tuple(signature))    # store the signature
                        self._forest_weights.append(quality)  # store rule weight

        return True if len(self._forest_signatures) > length else False

    def node_filter(self, j, index):
        _tree = self._estimators[index]
        _feature = _tree.feature
        _threshold = _tree.threshold
        formula = [[None for _ in range(self.n_features)] for _ in range(2)]  # store formula
        visited = np.zeros([2, self.n_features], dtype=float)  # record visiting state

        rule = self._forest_paths[index][j]
        for k, node in enumerate(rule[:-1]):
            ig = self._ig[index][j][k]

            if ig >= self._theta:       # node filter: ig >= theta
                f = _feature[node]
                if _tree.children_left[node] == rule[k + 1]:
                    if visited[0, f] == 0:
                        visited[0, f] = 1
                        formula[0][f] = _threshold[node]
                    else:
                        formula[0][f] = sub_f(_threshold[node], formula[0][f])
                else:
                    if visited[1, f] == 0:
                        visited[1, f] = 1
                        formula[1][f] = _threshold[node]
                    else:
                        formula[1][f] = add_f(_threshold[node], formula[1][f])

        if not np.all(visited == 0):  # store the rule only if there are nodes other than leaf
            self._forest_values.append(_tree.value[rule[-1]])  # store leaf votes
            self._forest_formulae.append(formula)
            self._forest_formulae_visited.append(visited)
            return 1
        else:
            return 0
