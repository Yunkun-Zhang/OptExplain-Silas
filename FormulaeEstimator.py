import numpy as np
from silas import add_f, sub_f
from tqdm import tqdm


class FormulaeEstimator(object):
    def __init__(self, z3processor, conjunction=False, classes=None):
        self._z3processor = z3processor
        self._groups = self._z3processor.groups_after_filter
        self.group_weights = self._z3processor.group_weights
        self._n_features = self._z3processor.n_features
        self.groups_signature = [t[0] for t in self._groups]
        self.groups_formula = []    # store formula combined into groups
                                    # each element with shape (2, n_features)
        self.visited = []
        self.sat_group = []
        self._conjunction = conjunction
        self._classes = classes
        self.scale = 0
        self.values = self._z3processor._extractor._estimators.nominal_features

    def _get_formulae(self):
        if self._conjunction is True:
            for group in self._groups:
                rules = group[1]
                formula = [[None for _ in range(self._n_features)] for _ in range(2)]
                visited = np.zeros([2, self._n_features])
                n = 1
                for rule in rules:
                    o_formula = self._z3processor.formulae[rule]
                    o_visited = self._z3processor.visited[rule]
                    if np.sum(o_visited) == 0:
                        print('Invalid rule existing')
                    n += 1
                    for feature in range(self._n_features):
                        if o_visited[0][feature] == 1:
                            if visited[0][feature] == 0:
                                formula[0][feature] = o_formula[0][feature]
                                visited[0][feature] = 1
                            else:
                                # min or intersection
                                formula[0][feature] = sub_f(formula[0][feature], o_formula[0][feature])
                        if o_visited[1][feature] == 1:
                            if visited[1][feature] == 0:
                                formula[1][feature] = o_formula[1][feature]
                                visited[1][feature] = 1
                            else:
                                # max or union
                                formula[1][feature] = add_f(formula[1][feature], o_formula[1][feature])
                self.groups_formula.append(formula)
                self.visited.append(visited)

        # compute #conjuncts
        o_visited = self._z3processor.visited
        for index, group in enumerate(self._groups):
            for r, rule in enumerate(group[1]):
                for i in range(self._n_features):
                    if o_visited[rule][0][i] == 1:
                        self.scale += 1
                    if o_visited[rule][1][i] == 1:
                        self.scale += 1

    def get_formulae_text(self, file):
        if self._conjunction is True:
            if len(self.groups_formula) == 0:
                self._get_formulae()

            for index, formula in enumerate(self.groups_formula):
                text = ''
                for i in range(self._n_features):
                    if self.visited[index][0][i] == 1:
                        f = formula[0][i]
                        if f.type == 'numeric':
                            text += f'(feature_{i} <= {f.value}) ∧ '
                        else:
                            text += f'(feature_{i} in ' \
                                    f'{[self.values[i][j] for j in range(len(self.values)) if f.value[j]]}) ∧ '
                    if self.visited[index][1][i] == 1:
                        f = formula[1][i]
                        if f.type == 'numeric':
                            text += f'(feature_{i} > {f.value}) ∧ '
                        else:
                            text += f'(feature_{i} not in ' \
                                    f'{[self.values[i][j] for j in range(len(self.values)) if f.value[j]]}) ∧ '
                text = text[:-2]
                title = f'Group {index:>2}: | {sum(self.group_weights[index]):>3.0f} samples | ' \
                        f'{len(self._groups[index][1]):>2} rules | {self._groups[index][0]}'
                print(title)
                print(text)
                file.write(title + '\n')
                file.write(text + '\n')
        else:
            self._get_formulae()
            o_formulae = self._z3processor.formulae
            o_visited = self._z3processor.visited
            for index, group in enumerate(self._groups):
                text = ''
                for r, rule in enumerate(group[1]):
                    text += f'{self.group_weights[index][r]:>15.0f} samples\t'
                    for i in range(self._n_features):
                        if o_visited[rule][0][i] == 1:
                            f = o_formulae[rule][0][i]
                            if f.type == 'numeric':
                                text += f'(feature_{i} <= {f.value}) ∧ '
                            else:
                                text += f'(feature_{i} in ' \
                                        f'{[self.values[i][j] for j in range(len(self.values)) if f.value[j]]}) ∧ '
                        if o_visited[rule][1][i] == 1:
                            f = o_formulae[rule][1][i]
                            if f.type == 'numeric':
                                text += f'(feature_{i} > {f.value}) ∧ '
                            else:
                                text += f'(feature_{i} not in ' \
                                        f'{[self.values[i][j] for j in range(len(self.values)) if f.value[j]]}) ∧ '
                    text = text[:-3] + '\n'

                title = f'Group {index:>2}: | {sum(self.group_weights[index]):>3.0f} samples | ' \
                        f'{len(self._groups[index][1]):>2} rules | {self._groups[index][0]}'
                print(title)
                print(text)
                file.write(title + '\n')
                file.write(text + '\n')
            print('conjuncts num:', self.scale)
            file.write("conjuncts num:" + str(self.scale) + '\n')

    def classify_a_sample(self, x):
        """Predict one instance by decision rules."""
        ans = np.zeros([len(self.groups_signature[0])])
        sat_g = []
        if self._conjunction is True:
            for index, formula in enumerate(self.groups_formula):
                sat = True
                for feature in range(self._n_features):
                    if self.visited[index][0][feature] == 1 and \
                            not formula[0][feature].satisfy(x[feature], self.values[feature]):
                        sat = False
                        break
                    if self.visited[index][1][feature] == 1 and \
                            formula[1][feature].satisfy(x[feature], self.values[feature]):
                        sat = False
                        break
                if sat is True:
                    ans += np.array(self.groups_signature[index]) * sum(self.group_weights[index])
                    sat_g.append(index)
        else:
            o_formulae = self._z3processor.formulae
            o_visited = self._z3processor.visited
            for index, group in enumerate(self._groups):
                sample_num = 0
                for r, rule in enumerate(group[1]):
                    sat = True
                    for feature in range(self._n_features):
                        if o_visited[rule][0][feature] == 1 and \
                                not o_formulae[rule][0][feature].satisfy(x[feature], self.values[feature]):
                            sat = False
                            break
                        if o_visited[rule][1][feature] == 1 and \
                                o_formulae[rule][1][feature].satisfy(x[feature], self.values[feature]):
                            sat = False
                            break
                    if sat is True:
                        sample_num += self.group_weights[index][r]
                if sample_num > 0:
                    ans += np.array(self.groups_signature[index]) * sample_num
                    sat_g.append(index)
        self.sat_group.append(sat_g)
        return ans

    def classify_samples_values(self, x):
        if len(self.groups_formula) == 0 and self._conjunction is True:
            self._get_formulae()

        ans = np.zeros((len(x), len(self._classes)))
        for index, sample in enumerate(x):
            cls = self.classify_a_sample(sample)

            if np.sum(cls == 0) == len(cls):
                ans[index] = [-1] * len(self._classes)
            else:
                ans[index] = cls
        return ans

    def classify_samples(self, x):
        """Predict the instances by decision rules."""
        if len(self.groups_formula) == 0 and self._conjunction is True:
            self._get_formulae()
        ans = []
        for sample in tqdm(x):
            cls = self.classify_a_sample(sample)

            if np.sum(cls == 0) == len(cls):
                ans.append(-1)
            else:
                ans.append(self._classes[cls.argmax()])
        return ans
