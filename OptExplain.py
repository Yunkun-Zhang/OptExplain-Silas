import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from silas import RFC
from Main_Process import MainProcess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', default='model/diabetes/',
                        help='root of your Silas model')
    parser.add_argument('-t', '--test-file', default='tests/clean-diabetes_test.csv',
                        help='path to your test file')
    parser.add_argument('-p', '--prediction-file', default='tests/predictions_diabetes.csv',
                        help='path to Silas-generated predictions.csv')
    parser.add_argument('--generation', type=int, default=20,
                        help='number of PSO iterations')
    parser.add_argument('--scale', type=int, default=20,
                        help='number of PSO particles')
    parser.add_argument('--acc-weight', type=float, default=0.5,
                        help='proportion of current acc in fitness computation')
    parser.add_argument('--conjunction', action='store_true',
                        help='whether to output conjunction')
    parser.add_argument('--max-sat', action='store_true',
                        help='whether to apply MAX-SAT')
    parser.add_argument('--no-tailor', action='store_true',
                        help='not to use size filter')
    args = vars(parser.parse_args())

    model_path = args['model_path']
    test_file = args['test_file']
    pf = args['prediction_file']

    # n_estimators = 100  # number of trees
    # max_depth = 10  # max-depth of each tree

    generation = args['generation']
    scale = args['scale']
    acc_weight = args['acc_weight']
    conjunction = args['conjunction']
    maxsat_on = args['max_sat']
    size_filter = not args['no_tailor']

    # read the test data
    test_data = pd.read_csv(test_file)
    X_test = np.array(test_data.iloc[:, 0:-1])
    y_test = np.array(test_data.iloc[:, -1])

    print('RF...')
    clf = RFC(model_path, pf)
    print('RF acc:', accuracy_score(y_test, clf.predict()))

    # output
    file_num = 1
    if not os.path.exists('explanation'):
        os.makedirs('explanation')
    while os.path.exists(f'explanation/{file_num}.txt') is True:
        file_num += 1
    file = open(f'explanation/{file_num}.txt', 'w')
    # file.write(f'n_estimators = {n_estimators}\tmax_depth = {max_depth}\n')
    file.write('generation = {}\tscale = {}\tacc_weight = {}\tmaxsat = {}\ttailor = {}\n'.
               format(generation, scale, acc_weight, maxsat_on, size_filter))
    print('explain...')
    file.write('begin\n')
    m = MainProcess(clf, X_test, y_test, file, generation=generation, scale=scale, acc_weight=acc_weight,
                    conjunction=conjunction, maxsat_on=maxsat_on, tailor=size_filter, fitness_func='Opt')
    best_param = m.pso()
    # best_param = [0.5, 0.6, 0.5, -1]
    m.explain(best_param, auc_plot=False)
    file.write('end')
    file.close()
