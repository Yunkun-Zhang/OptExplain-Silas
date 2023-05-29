import json
import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict

__version__ = '0.8.7'


class Feature:
    """Process numerical and nominal features."""

    def __init__(self, value: Union[float, List[bool]]) -> None:
        self.value = value
        self.type = 'numeric' if isinstance(value, (int, float)) else 'nominal'

    def __repr__(self):
        return str(self.value)

    def satisfy(self, f: Union[float, str], values: List[str] = None) -> bool:
        """Test if the input satisfies the feature.

        Args:
          f: A feature value.
          values: List of nominal feature values.

        Returns:
          f <= feature or f ∈ feature.
        """
        if self.type == 'numeric':
            return float(f) <= float(self.value)
        else:
            if float(f) in values:
                return self.value[values.index(float(f))]
            elif int(float(f)) in values:
                return self.value[values.index(int(float(f)))]
            elif str(float(f)) in values:
                return self.value[values.index(str(float(f)))]
            elif str(int(float(f))) in values:
                return self.value[values.index(str(int(float(f))))]
            else:
                raise ValueError('The type of f does not match that of values.')


def add_f(f1: Feature, f2: Feature) -> Feature:
    """Compute maximum of numerical features or union of nominal features."""
    if f1.type == f2.type == 'numeric':
        return Feature(max(f1.value, f2.value))
    return Feature(np.bitwise_or(f1.value, f2.value))


def sub_f(f1: Feature, f2: Feature) -> Feature:
    """Compute minimum of numerical features or intersection of nominal features."""
    if f1.type == f2.type == 'numeric':
        return Feature(min(f1.value, f2.value))
    return Feature(np.bitwise_and(f1.value, f2.value))


class DT:
    """Decision tree."""

    def __init__(self, n_classes_: int = None):
        self.n_classes_ = n_classes_
        self.oob_score = 0
        self.count = 0              # number of nodes
        self.feature = []           # feature indices at each node
        self.rules = []             # leaf paths
        self.children_left = []
        self.children_right = []
        self.value = []             # number of samples at each node
        self.ig = []                # information gain at each node
        self.threshold = []         # feature threshold or partition at each node

    def initialize(self, weight: float, dic: Dict, index: List[int]) -> None:
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
                # calculate entropy
                entropy = 0
                for p in prob:
                    if p != 0:
                        entropy -= p * np.log2(p)
                self.ig[number] = entropy
            elif 'value' in d:
                # one-hot value, entropy=0
                value = np.zeros(self.n_classes_ or 2)
                value[d['value']] = 1
            else:
                self.feature[number] = index[d['featureIndex']]
                if 'threshold' in d:
                    self.children_left[number], value1 = dfs(d['left'])
                    self.children_right[number], value2 = dfs(d['right'])
                else:
                    self.children_left[number], value1 = dfs(d['right'])
                    self.children_right[number], value2 = dfs(d['left'])
                value = value1 + value2
                self.ig[number] = d.get('weight') or .1
                self.threshold[number] = Feature(d['threshold'] if 'threshold' in d else d['partition'])
            rule.pop()
            self.value[number] = value
            return number, value

        dfs(dic)

    def __getitem__(self, item: int) -> List[int]:
        return self.rules[item]


class RFC:
    """Random forest classifier."""

    def __init__(self, model_path='', pred_file='predictions.csv', label_column=None) -> None:
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

        if label_column is None:
            label_column = len(metadata['features'])
        label = metadata['features'][label_column]['feature-name']

        # for nominal feature method 'satisfy'
        dic = {f['name']: f for f in metadata['attributes']}
        self.nominal_features = []
        for f in metadata['features']:
            if f['attribute-name'] == label:
                continue
            if 'values' in dic[f['attribute-name']]:
                self.nominal_features.append(dic[f['attribute-name']]['values'])
            else:
                self.nominal_features.append(None)

        # features are ordered according to metadata
        features = [a['feature-name'] for a in metadata['features']]
        label_column = features.index(label)
        self.features_ = features[:label_column] + features[label_column + 1:]
        self.output_feature_ = label

        # revise featureIndex in tree nodes
        self.feature_indices = [self.features_.index(f) for f in summary['template']]

        self.n_estimators = summary['size']
        self.classes_ = summary['output-labels']
        self.n_classes_ = len(self.classes_)
        self.n_features_ = len(self.features_)
        self.n_outputs_ = 1
        self.trees_ = []
        self.trees_oob_scores = []

        self._set_oob_scores(summary['trees'])  # read tree oob scores
        self._build_trees(summary['trees'])  # build decision trees

    def __getitem__(self, item: int) -> DT:
        return self.trees_[item]

    def _set_oob_scores(self, trees_dic):
        for tree in trees_dic:
            # in Silas v0.8.7 this is the oob score
            self.trees_oob_scores.append(tree['weight'])

    def _build_trees(self, trees_dic):
        for tree in trees_dic:
            with open(os.path.join(self.model_path, tree['path']), 'r') as f:
                d = json.load(f)
            dt = DT()
            dt.initialize(tree['weight'], d, self.feature_indices)
            self.trees_.append(dt)

    def predict_proba(self):
        """Predict probability of each class for each input instance."""
        prob = pd.read_csv(self.pred_file).values.tolist()
        return np.array(prob)

    def predict(self, use_cls_names=False):
        """Predict class for each input instance."""
        prob = self.predict_proba()
        cls = np.argmax(prob, axis=-1)
        if use_cls_names:
            cls = [self.classes_[c] for c in cls]
        return np.array(cls)
