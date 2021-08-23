import json
import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict

__version__ = '0.8.7'


class Feature:
    """Process numerical and nominal features."""

    def __init__(self, f: Union[float, List[bool]]) -> None:
        self.f = f
        self.type = 'numeric' if isinstance(f, (int, float)) else 'nominal'

    def __repr__(self):
        return str(self.f)

    def satisfy(self, f: Union[float, str], values: List[str] = None) -> bool:
        """Test if the input satisfies the feature.

        Args:
          f: A float or a partition.
          values: List of nominal feature values.

        Returns:
          f <= feature or f ∈ feature.
        """
        if self.type == 'numeric':
            return f <= self.f
        else:
            return self.f[values.index(f)]


def add_f(f1: Feature, f2: Feature) -> Feature:
    """Compute maximum of numerical features or union of nominal features."""
    if f1.type == f2.type == 'numeric':
        return Feature(max(f1.f, f2.f))
    return Feature(np.bitwise_or(f1.f, f2.f))


def sub_f(f1: Feature, f2: Feature) -> Feature:
    """Compute minimum of numerical features or intersection of nominal features."""
    if f1.type == f2.type == 'numeric':
        return Feature(min(f1.f, f2.f))
    return Feature(np.bitwise_and(f1.f, f2.f))


class DT:
    """Decision tree."""

    def __init__(self):
        self.oob_score = 0
        self.count = 0              # number of nodes
        self.feature = []           # feature indices at each node
        self.rules = []             # leaf paths
        self.children_left = []
        self.children_right = []
        self.value = []             # number of samples at each node
        self.ig = []                # information gain at each node
        self.threshold = []         # feature threshold or partition at each node

    def initialize(self, weight: float, dic: Dict) -> None:
        """Initialize a decision tree with Silas-generated .json file."""
        self.__init__()
        self.oob_score = weight
        rule = []

        def dfs(d):
            number = self.count
            self.count += 1
            rule.append(number)
            self.feature.append(-1)
            self.children_left.append(-1)
            self.children_right.append(-1)
            self.value.append([])
            self.ig.append(0)
            self.threshold.append(None)
            if 'aggregate' in d:
                self.rules.append(rule[:])
                value = np.array(d['aggregate'])
                prob = value / value.sum()
                self.ig[number] = -(prob * np.log2(prob)).sum() if prob[0] != 0 and prob[1] != 0 else 0
            else:
                self.feature[number] = d['featureIndex']
                self.children_left[number], value1 = dfs(d['left'])
                self.children_right[number], value2 = dfs(d['right'])
                value = value1 + value2
                self.ig[number] = d['weight']
                self.threshold[number] = Feature(d['threshold'] if 'threshold' in d else d['partition'])
            rule.pop()
            self.value[number] = value
            return number, value

        dfs(dic)

    def __getitem__(self, item: int) -> List[int]:
        return self.rules[item]


class RFC:
    """Random forest classifier."""

    def __init__(self, model_path='', pred_file='predictions.csv') -> None:
        """Build a random forest classifier.

        Args:
          model_path: Path to Silas model.
          pred_file: Path of Silas prediction file.
        """
        self.model_path = model_path
        self.pred_file = pred_file

        with open(os.path.join(model_path, 'summary.json'), 'r') as f:
            summary = json.load(f)
        with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        dic = {f['name']: f for f in metadata['attributes']}
        self.nominal_features = [dic[f['attribute-name']]['values'] if 'values' in f else None
                                 for f in metadata['features']]  # for nominal feature method 'satisfy'
        self.n_estimators = summary['size']
        self.classes_ = summary['output-labels']

        features = [a['feature-name'] for a in metadata['features']]
        self.features_ = features[:-1]
        self.output_feature_ = features[-1]
        self.n_classes_ = len(summary['output-labels'])
        self.n_features_ = len(summary['template'])
        self.n_outputs_ = 1
        self.trees_dic = summary['trees']
        self.trees_ = []
        self.trees_oob_scores = []

        self._set_oob_scores()  # read tree oob scores
        self._build_trees()  # build decision trees

    def __getitem__(self, item: int) -> DT:
        return self.trees_[item]

    def _set_oob_scores(self):
        for tree in self.trees_dic:
            # in Silas v0.8.7 this is the oob score
            self.trees_oob_scores.append(tree['weight'])

    def _build_trees(self):
        for tree in self.trees_dic:
            with open(os.path.join(self.model_path, tree['path']), 'r') as f:
                d = json.load(f)
            dt = DT()
            dt.initialize(tree['weight'], d)
            self.trees_.append(dt)

    def predict_proba(self):
        """Predict probability of each class for each input instance."""
        prob = pd.read_csv(self.pred_file).values.tolist()
        return np.array(prob)

    def predict(self):
        """Predict class for each input instance."""
        prob = self.predict_proba()
        pred = [self.classes_[0] if x[0] > x[1] else self.classes_[1] for x in prob]
        return np.array(pred)
