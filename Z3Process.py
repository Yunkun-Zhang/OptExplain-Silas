from z3 import *
import math


class Z3Process(object):
    def __init__(self, extractor=None, k=-1):
        # k <= 0:    all rules
        # 0 < k < 1: k of all rules (ratio)
        # k >= 1:    k rules
        self._extractor = extractor
        self._groups = []   # [G1, G2, ..., Gk]
                            # Gi: [(signature), [rule indices satisfying signature],
                            #      [rule indices after MAX-SAT]]
        self._signatures = self._extractor._forest_signatures
        self.formulae = self._extractor._forest_formulae
        self.visited = self._extractor._forest_formulae_visited
        self.n_features = self._extractor.n_features
        self._values = self._extractor._forest_values
        self.group_weights = []
        self._k = k
        self.groups_after_filter = []
        self.n_rules_after_max = 0
        self.n_rules_after_filter = 0

    def leaves_partition(self):
        for index, signature in enumerate(self._signatures):
            tag = 0
            for i in range(len(self._groups)):
                if signature in self._groups[i]:
                    self._groups[i][1].append(index)
                    tag = 1
                    break
            if tag == 0:
                self._groups.append([signature, [index]])

    def maxsat(self):
        for i in range(self.n_features):
            exec('feature_{} = Real(\'feature_{}\')'.format(i, i))
        for index, group in enumerate(self._groups):
            opt = Optimize()
            for rule_num in group[1]:
                rule_name = 'rule_' + str(rule_num)
                exec('{} = Bool(\'{}\')'.format(rule_name, rule_name))
                text = rule_name + ' == ' + self.formulae_text(rule_num)
                # print(text, self._values[rule_num].sum())
                opt.add(eval(text))
                opt.add_soft(eval(rule_name), self._values[rule_num].sum())  # use leaf vote for MAX-SAT weight
            opt.check()
            m = opt.model()
            tmp = []
            for decl in m:
                if isinstance(m[decl], z3.z3.BoolRef) and m[decl] == True:  # do not delete "== True"
                    tmp.append(int(str(decl)[5:]))
            self._groups[index].append(tmp)
            # print(self._groups[index])
            # print(len(self._groups[index][1]), len(self._groups[index][2]))
            value_sum1 = 0          # #samples before MAX-SAT
            value_sum2 = 0          # #samples after MAX-SAT
            for rule_num in self._groups[index][1]:
                value_sum1 += self._values[rule_num]
            for rule_num in self._groups[index][2]:
                value_sum2 += self._values[rule_num]
            self.n_rules_after_max += len(self._groups[index][2])

    def formulae_text(self, index):
        text = 'And('
        for i in range(self.n_features):
            if self.visited[index][0][i] == 1:
                text += 'feature_' + str(i) + ' <= ' + str(self.formulae[index][0][i]) + ', '
            if self.visited[index][1][i] == 1:
                text += 'feature_' + str(i) + ' > ' + str(self.formulae[index][1][i]) + ', '
        return text[:-2] + ')'

    def filter(self, _k):
        groups = []
        for i in range(len(self._groups)):
            _group = [self._groups[i][0]]
            if _k <= 0 or _k >= len(self._groups[i][-1]):
                k_to_num = len(self._groups[i][-1])
            elif _k < 1:
                k_to_num = math.ceil(len(self._groups[i][-1]) * _k)
            else:
                k_to_num = math.floor(_k)
            # self._groups[i][2].sort(key=lambda x: self._values[x])
            # self._groups[i][2] = self._groups[i][2][:k_to_num]
            _group.append(sorted(self._groups[i][-1], key=lambda x: sum(self._values[x]), reverse=True)[:k_to_num])
            groups.append(_group)
            self.n_rules_after_filter += k_to_num
        return groups

    def run_filter(self):
        self.groups_after_filter = self.filter(self._k)
        self.group_weights = []
        for _group in self.groups_after_filter:
            _group_weight = []
            for rule_num in _group[1]:
                _group_weight.append(sum(self._values[rule_num]))
            self.group_weights.append(_group_weight)

    def set_k(self, k):
        self._k = k
