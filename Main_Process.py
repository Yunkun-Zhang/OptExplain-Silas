from Extractor import Extractor
from Z3Process import Z3Process
from FormulaeEstimator import FormulaeEstimator
import numpy as np
import copy
from time import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import traceback
from functools import partial
from sklearn.metrics import accuracy_score


class Log:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
        except Exception as e:
            multiprocessing.get_logger().error(traceback.format_exc())
            raise
        return result


class MainProcess(object):
    def __init__(self, clf, x_test, y_test, file, generation=10, scale=10, conjunction=False, acc_weight=0.5,
                 maxsat_on=True, tailor=True, fitness_func='Opt'):
        self._clf = clf                 # random forest
        self._X_test = x_test
        self._y_test = y_test
        self._nfeature = x_test.shape[1]
        self._acc_weight = acc_weight   # weight of acc in fitness function
        self._scale = scale             # number of particles
        self._generation = generation   # number of iterations
        self._conjunction = conjunction
        self._file = file
        self._maxsat_on = maxsat_on
        self._tailor = tailor
        self._quality = []
        self._ig = []
        self.fitness_func = fitness_func

    def pso_function(self, extractor, params, _rf_res):
        offset = []
        r_num = []
        for each in params.T:
            ex = copy.deepcopy(extractor)
            _offset = 0
            phi = each[0]
            theta = each[1]
            psi = each[2]
            k = each[3]
            ex.set_param(phi, theta, psi)
            ex.rule_filter()
            if len(ex._forest_values) == 0:
                offset.append(-1)
                r_num.append(1)
                continue
            sat = Z3Process(ex, k)
            sat.leaves_partition()
            if self._maxsat_on is True:
                sat.maxsat()
            sat.run_filter()
            f = FormulaeEstimator(sat, conjunction=self._conjunction, classes=self._clf.classes_)
            f._get_formulae()
            res = f.classify_samples(self._X_test)
            for index in range(len(res)):
                if res[index] == _rf_res[index]:
                    _offset += 1
            offset.append(_offset)
            r_num.append(sat.n_rules_after_filter)
        return np.array(offset), np.array(r_num)

    def pso_function_parallel(self, extractor, params, _rf_res):
        pool = Pool()
        time1 = time()
        func = partial(func_parallel, extractor, _rf_res, self._maxsat_on, self._conjunction, self._clf.classes_,
                       self._X_test, self._quality, self._ig, self.fitness_func)
        time2 = time()
        res = pool.map(Log(func), params.T.tolist())
        time3 = time()
        pool.close()
        pool.join()
        time4 = time()
        res = np.array(res)
        offset = res[:, 0]
        r_num = res[:, 1]
        return np.array(offset), np.array(r_num)

    def pso(self):
        start_pso = time()
        np.set_printoptions(precision=3)
        print('------------ P S O -------------')
        self._file.write('------------ P S O -------------\n')
        ex = Extractor(self._clf)
        ex.extract_forest_paths()

        # ex.count_quality()
        #
        # self._quality, self._ig = ex.opt_get_quality()
        # ex.opt_clear_quality()

        RF_res = self._clf.predict()
        sample_num = len(self._y_test) if self.fitness_func == 'Opt' else 1
        w_max = 0.9
        w_min = 0.4
        c1, c2 = 1.6, 1.6  # learn factors
        max_gen = self._generation
        sizepop = self._scale

        # v_min = [-0.1, -0.1, -0.1, -3]
        # v_max = [0.1, 0.1, 0.1, 3]
        # pop_min = [0, 0, 0.1, -2]
        # pop_max = [1, 1, 1, 30]

        # speed constraints
        v_min = [-0.1, -0.1, -0.1, -3]
        v_max = [0.1, 0.1, 0.1, 3]
        # parameter constraints
        pop_min = [0, 0, 0, 1]
        pop_max = [1, 1, 1, 30]

        # initialize the parameters
        np.random.seed(10)
        pop = np.zeros([4, sizepop])
        pop[0] = np.random.uniform(0.1, 1, (1, sizepop))  # phi
        pop[1] = np.random.uniform(0.1, 1, (1, sizepop))  # theta
        pop[2] = np.random.uniform(0, 1, (1, sizepop))    # psi
        pop[3] = np.random.uniform(1, 30, (1, sizepop))   # k
        if self._tailor is False:
            pop[3] = -1

        # initialize speed
        v = np.random.uniform(-0.1, 0.1, (4, sizepop)) * [[1], [1], [1], [30]]

        # do pso
        offset, r_num = self.pso_function_parallel(ex, pop, RF_res)

        # compute fitness
        if self.fitness_func == 'Pro':
            fitness = self.pro_fitness(offset, r_num)
        else:
            fitness = self.opt_fitness(offset, r_num)

        # best particle
        i = np.argmax(fitness)

        g_best = pop                # best parameters of each particle
        z_best = pop[:, i]          # best parameters
        fitness_gbest = fitness     # best fitness of each particle
        fitness_zbest = fitness[i]  # best fitness
        print(0, '\t', i, '\t', pop[:, i], '\t', offset[i] / sample_num, r_num[i], 'fitness:', fitness[i])
        self._file.write(
            '0:\t{}\t{}\t{:.2f}\t{} fitness: {:.4f}\n'.format(i, pop[:, i], (offset[i] / sample_num), r_num[i],
                                                              fitness[i]))

        record = np.zeros(max_gen)  # store best fitness
        for t in range(max_gen):
            # update inertia
            w = w_max - (w_max - w_min) / max_gen * t

            # update speed
            v = w * v + c1 * np.random.random() * (g_best - pop) + c2 * np.random.random() * \
                (z_best.reshape(4, 1) - pop)
            for i in range(4):
                v[i][v[i] > v_max[i]] = v_max[i]
                v[i][v[i] < v_min[i]] = v_min[i]

            # update parameters
            pop = pop + v
            for i in range(4):
                pop[i][pop[i] > pop_max[i]] = pop_max[i]
                pop[i][pop[i] < pop_min[i]] = pop_min[i]

            if self._tailor is False:
                pop[3] = -1

            # iteration
            offset, r_num = self.pso_function_parallel(ex, pop, RF_res)

            # compute fitness
            if self.fitness_func == 'Pro':  # 计算适应度
                fitness = self.pro_fitness(offset, r_num)
            else:
                fitness = self.opt_fitness(offset, r_num)

            # update best positions
            index = fitness > fitness_gbest
            fitness_gbest[index] = fitness[index]
            g_best[:, index] = pop[:, index]

            # best particle
            j = np.argmax(fitness)

            print(t + 1, '\t', j, '\t', pop[:, j], '\t', offset[j] / sample_num, r_num[j], 'fitness:', fitness[j])
            self._file.write(
                '{}:\t{}\t{}\t{:.2f}\t{} fitness: {:.2f}'.format(t + 1, j, pop[:, j], offset[j] / sample_num, r_num[j],
                                                                 fitness[j]))
            if fitness[j] > fitness_zbest:
                z_best = pop[:, j]
                fitness_zbest = fitness[j]
                print('new record: ', fitness[j])
                self._file.write('\t*new record*')
            self._file.write('\n')
            record[t] = fitness_zbest  # store best fitness

        print('optimal parameters', z_best)
        self._file.write('optimal parameters: {}\n'.format(z_best))

        end_pso = time()
        print('pso time:', end_pso - start_pso)
        self._file.write('pso time: {}\n\n'.format(end_pso - start_pso))

        return z_best

    def opt_fitness(self, offset, r_num):
        # Eq (9)
        acc = offset / len(self._X_test)
        # fo = self._acc_weight/(1 + np.exp(-10 * (acc - 0.5)))
        fo = self._acc_weight * acc
        fr_sub = np.exp(-5 * (r_num / (self._nfeature * self._clf.n_classes_) - 1))
        fr = (1 - self._acc_weight) * fr_sub / (1 + fr_sub)
        return fo + fr

    def pro_fitness(self, g_num, r_num):
        # Eq (10)
        fitness = (self._clf.n_classes_ - g_num + 1) * r_num
        return fitness

    def explain(self, param, label='', auc_plot=False):
        print('------------ Explanation -------------')
        self._file.write('------------ Explanation -------------\n')
        phi = param[0]
        theta = param[1]
        psi = param[2]
        k = param[3]

        start1 = time()
        ex = Extractor(self._clf, phi, theta, psi)
        ex.extract_forest_paths()

        ex.rule_filter()

        print('max_rule', ex.max_rule, 'max_node', ex.max_node)
        print('min_rule', ex.min_rule, 'min_node', ex.min_node)
        end1 = time()
        print("EX Running time: %s seconds" % (end1 - start1))

        print("Original #rules:", ex.n_original_leaves_num)
        print("Original scale:", ex.scale)
        print("#rules after rule-filter:", len(ex._forest_values))
        self._file.write('Original #rules: {}\n'.format(ex.n_original_leaves_num))
        self._file.write('Original scale: {}\n'.format(ex.scale))
        self._file.write('#rules after rule-filter: {}\n'.format(len(ex._forest_values)))

        # do MAX-SAT
        start2 = time()
        sat = Z3Process(ex, k)
        sat.leaves_partition()
        if self._maxsat_on is True:
            sat.maxsat()
            text = f'#rules after MAX-SAT: {sat.n_rules_after_max}, after rule-filter: {sat.n_rules_after_filter}\n'
            print(text)
            self._file.write(text)
        else:
            print('no MAX-SAT')
            self._file.write('/no MAX-SAT\n')
        sat.run_filter()
        end2 = time()
        print(f"MAX-SAT running time: {end2 - start2} seconds")

        print('classes:', self._clf.classes_)

        # get formulae
        start3 = time()
        f = FormulaeEstimator(sat, conjunction=self._conjunction, classes=self._clf.classes_)
        f.get_formulae_text(self._file)

        # evaluation
        print('\n------------ Performance -------------')
        self._file.write('\n------------ Performance -------------\n')
        c_ans = self._clf.predict()
        ans = f.classify_samples(self._X_test)
        end3 = time()
        print(f"ET Running time: {end3 - start3} seconds")

        RF_accuracy = accuracy_score(self._y_test, c_ans)
        EX_accuracy = accuracy_score(self._y_test, ans)
        performance = accuracy_score(c_ans, ans)

        no_ans = 0
        overlap = 0
        for each in f.sat_group:
            if len(each) > 1:
                overlap += 1
            elif len(each) == 0:
                no_ans += 1

        if label == '':
            label = self._clf.classes_[1]  # true

        fpr, tpr, thresholds = roc_curve(self._y_test, self._clf.predict_proba()[:, 1], pos_label=label)
        ori_auc = auc(fpr, tpr)

        ex_test = f.classify_samples_values(self._X_test)
        efpr, etpr, ethresholds = roc_curve(self._y_test, ex_test[:, 1], pos_label=label)
        ex_auc = auc(efpr, etpr)

        print('sample size:\t', len(self._y_test))
        self._file.write('sample size:\t{}\n'.format(len(self._y_test)))

        print('RF accuracy:\t', RF_accuracy)
        self._file.write('RF accuracy:\t{}\n'.format(RF_accuracy))

        print('RF AUC:\t\t\t', ori_auc)
        self._file.write('RF AUC:\t\t\t{:.2f}\n'.format(ori_auc))

        print('EX accuracy:\t', EX_accuracy)
        self._file.write('EX accuracy:\t{}\n'.format(EX_accuracy))

        print('EX AUC:\t\t\t', ex_auc)
        self._file.write('EX AUC:\t\t\t{:.2f}\n'.format(ex_auc))

        print('Coverage:\t\t', (len(self._y_test) - no_ans) / len(self._y_test))
        self._file.write('Coverage:\t\t{}\n'.format((len(self._y_test) - no_ans) / len(self._y_test)))

        print('Overlap:\t\t', overlap / len(self._y_test))
        self._file.write('Overlap:\t\t{}\n'.format(overlap / len(self._y_test)))

        print('*Performance:\t', performance)
        self._file.write('*Performance:\t{}\n'.format(performance))

        # plot the results
        if auc_plot is True:
            plt.plot(fpr, tpr, linewidth=2, label="RF ROC curve (area = {:.2f})".format(ori_auc))
            plt.plot(efpr, etpr, linewidth=2, label="Explain ROC curve (area = {:.2f})".format(ex_auc))
            plt.xlabel("false positive rate")
            plt.ylabel("true positive rate")
            plt.ylim(0, 1.05)
            plt.xlim(0, 1.05)
            plt.legend(loc=4)
            plt.show()


def func_parallel(extractor, _rf_res, maxsat_on, conjunction, classes, X_test, quality, ig, fitness_func, param):
    # t0 = time()
    # ex = copy.deepcopy(extractor)
    # ex.opt_set_quality(quality, ig)
    # t1 = time()         # plan A

    t0 = time()
    ex = copy.deepcopy(extractor)
    ex.count_quality()
    t1 = time()  # Plan B

    # set extractor parameters
    _offset = 0
    phi = param[0]
    theta = param[1]
    psi = param[2]
    k = param[3]
    ex.set_param(phi, theta, psi)

    t2 = time()
    tag = ex.rule_filter()
    if tag is False:
        return -1, 1
    t3 = time()

    # if len(ex._forest_values) == 0:
    #     return -1, 1
    sat = Z3Process(ex, k)
    sat.leaves_partition()
    if maxsat_on is True:
        sat.maxsat()
    sat.run_filter()
    t4 = time()

    f = FormulaeEstimator(sat, conjunction=conjunction, classes=classes)
    f._get_formulae()

    if fitness_func == 'Pro':
        return len(f.groups_signature), f.scale

    res = f.classify_samples(X_test)
    t5 = time()

    for index in range(len(res)):
        if res[index] == _rf_res[index]:
            _offset += 1
    t6 = time()

    return _offset, f.scale