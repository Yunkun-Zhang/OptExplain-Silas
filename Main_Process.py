from Extractor import Extractor
from Z3Process import Z3Process
from FormulaeEstimator import FormulaeEstimator
import numpy as np
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
    """Logger for multi-processing.

    Use pool.map(Log(func), **kwargs) and the detailed
    error message will be printed out.
    """

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
    """Perform main process."""

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
            ex = extractor.copy()
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
        """Perform main process."""
        start_pso = time()
        np.set_printoptions(precision=3)
        print('---------------- P S O -----------------')
        self._file.write('---------------- P S O -----------------\n')

        # initialize extractor
        ex = Extractor(self._clf)
        ex.count_quality()

        # some fixed parameters
        RF_res = self._clf.predict()
        sample_num = len(self._y_test) if self.fitness_func == 'Opt' else 1
        w_max = 0.9  # inertia decrease from this
        w_min = 0.4  # to this linearly
        c1, c2 = 1.6, 1.6  # learning factors
        max_gen = self._generation
        sizepop = self._scale

        # speed constraints
        v_min = [-0.1, -0.1, -0.1, -3]
        v_max = [0.1, 0.1, 0.1, 3]
        # parameter constraints
        pop_min = [0, 0, 0, 1]
        pop_max = [1, 1, 1, 30]

        # initialize the parameters
        np.random.seed(10)
        pop = np.zeros([4, sizepop])
        max_phi = ex.max_rule
        max_theta = ex.max_node
        pop[0] = np.random.uniform(0, max_phi, (1, sizepop))    # phi
        pop[1] = np.random.uniform(0, max_theta, (1, sizepop))  # theta
        pop[2] = np.random.uniform(0, 1, (1, sizepop))          # psi
        pop[3] = np.random.uniform(1, 30, (1, sizepop))         # k
        if self._tailor is False:
            pop[3] = -1

        # initialize speed
        v = np.random.uniform(-0.1, 0.1, (4, sizepop)) * [[1], [1], [1], [30]]

        # do pso
        # offset, r_num = self.pso_function_parallel(ex, pop, RF_res)  # parallel
        offset, r_num = self.pso_function(ex, pop, RF_res)  # not parallel

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
        text = f'{0:<4}{i:<4}{pop[:, i]}\t{offset[i] / sample_num:>11.8f}  {r_num[i]:<2}  fitness: {fitness[i]:.8f}'
        print(text)
        self._file.write(text + '\n')

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
            # offset, r_num = self.pso_function_parallel(ex, pop, RF_res)  # parallel
            offset, r_num = self.pso_function(ex, pop, RF_res)  # not parallel

            # compute fitness
            if self.fitness_func == 'Pro':
                fitness = self.pro_fitness(offset, r_num)
            else:
                fitness = self.opt_fitness(offset, r_num)

            # update best positions
            index = fitness > fitness_gbest
            fitness_gbest[index] = fitness[index]
            g_best[:, index] = pop[:, index]

            # best particle
            j = np.argmax(fitness)

            text = f'{t + 1:<4}{j:<4}{pop[:, j]}\t{offset[j] / sample_num:>11.8f}  {r_num[j]:<2}  fitness: {fitness[j]:.8f}'
            print(text)
            self._file.write(text)
            if fitness[j] > fitness_zbest:
                z_best = pop[:, j]
                fitness_zbest = fitness[j]
                print('new record: ', fitness[j])
                self._file.write('\t*new record*')
            self._file.write('\n')
            record[t] = fitness_zbest  # store best fitness

        print(f'Optimal parameters: {z_best}')
        self._file.write('Optimal parameters: {}\n'.format(z_best))

        end_pso = time()
        print(f'PSO time: {end_pso - start_pso}\n')
        self._file.write('PSO time: {}\n\n'.format(end_pso - start_pso))

        return z_best

    def opt_fitness(self, offset, r_num):
        """Eq 9."""
        acc = offset / len(self._X_test)
        # fo = self._acc_weight/(1 + np.exp(-10 * (acc - 0.5)))
        fo = self._acc_weight * acc
        fr_sub = np.exp(-5 * (r_num / (self._nfeature * self._clf.n_classes_) - 1))
        fr = (1 - self._acc_weight) * fr_sub / (1 + fr_sub)
        return fo + fr

    def pro_fitness(self, g_num, r_num):
        """Eq 10."""
        fitness = (self._clf.n_classes_ - g_num + 1) * r_num
        return fitness

    def explain(self, param, label='', auc_plot=False):
        """Perform explanation."""
        print('---------------- Explanation -----------------')
        self._file.write('---------------- Explanation -----------------\n')
        phi = param[0]
        theta = param[1]
        psi = param[2]
        k = param[3]

        start1 = time()
        ex = Extractor(self._clf, phi, theta, psi)

        ex.rule_filter()

        print(f'max_rule {ex.max_rule:>8.6f}  max_node {ex.max_node:>8.6f}')
        print(f'max_rule {ex.min_rule:>8.6f}  max_node {ex.min_node:>8.6f}')
        end1 = time()
        print(f"EX Running time: {end1 - start1} seconds")

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
            self._file.write(text + '\n')
        else:
            print('No MAX-SAT')
            self._file.write('No MAX-SAT\n')
        sat.run_filter()
        end2 = time()
        print(f"SAT running time: {end2 - start2} seconds\n")

        print('Classes:', self._clf.classes_)

        # get formulae
        start3 = time()
        f = FormulaeEstimator(sat, conjunction=self._conjunction, classes=self._clf.classes_)
        f.get_formulae_text(self._file)

        # evaluation
        print('\n---------------- Performance -----------------')
        self._file.write('\n---------------- Performance -----------------\n')
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

        print(f'Sample size:     {len(self._y_test)}')
        self._file.write(f'Sample size:     {len(self._y_test)}\n')

        print(f'RF accuracy:     {RF_accuracy}')
        self._file.write(f'RF accuracy:     {RF_accuracy}\n')

        print(f'RF AUC:          {ori_auc}')
        self._file.write(f'RF AUC:          {ori_auc:.2f}\n')

        print(f'EX accuracy:     {EX_accuracy}')
        self._file.write(f'EX accuracy:     {EX_accuracy}\n')

        print(f'EX AUC:          {ex_auc}')
        self._file.write(f'EX AUC:          {ex_auc:.2f}\n')

        print(f'Coverage:        {(len(self._y_test) - no_ans) / len(self._y_test)}')
        self._file.write(f'Coverage:        {(len(self._y_test) - no_ans) / len(self._y_test)}\n')

        print(f'Overlap:         {overlap / len(self._y_test)}')
        self._file.write(f'Overlap:         {overlap / len(self._y_test)}\n')

        print(f'*Performance:    {performance}')
        self._file.write(f'*Performance:    {performance}\n')

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
    t0 = time()
    ex = extractor.copy()
    # ex.count_quality()
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
