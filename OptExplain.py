import os
from silas import RFC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import argparse
from Main_Process import MainProcess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', default='model/flights',
                        help='root of your Silas model')
    parser.add_argument('-t', '--test-file', default='tests/clean-flights_test.csv',
                        help='path to your test file')
    parser.add_argument('-p', '--prediction-file', default='tests/predictions_flights.csv',
                        help='path to Silas-generated predictions.csv')
    parser.add_argument('--number-of-trees', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    args = vars(parser.parse_args())

    model_path = args['model_path']
    test_file = args['test_file']
    pf = args['prediction_file']

    n_estimators = args['number_of_trees']  # number of trees
    max_depth = args['max_depth']  # max-depth of each tree

    generation = 20  # number of iterations
    scale = 20  # number of particles
    acc_weight = 0.5
    conjunction = False
    maxsat_on = False
    size_filter = True

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
    file.write(f'n_estimators = {n_estimators}\tmax_depth = {max_depth}\n')
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
