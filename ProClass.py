import os
import argparse
import numpy as np
import pandas as pd
from silas import RFC
from Main_Process import MainProcess
from Extractor import Extractor
from Z3Process import Z3Process
from FormulaeEstimator import FormulaeEstimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', default='model/diabetes/',
                        help='root of your Silas model')
    parser.add_argument('-t', '--test-file', default='tests/clean-diabetes_test.csv',
                        help='path to your test file')
    parser.add_argument('-p', '--prediction-file', default='tests/predictions_diabetes.csv',
                        help='path to Silas-generated predictions.csv')
    parser.add_argument('--number-of-trees', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    args = vars(parser.parse_args())

    model_path = args['model_path']
    test_file = args['test_file']
    pf = args['prediction_file']

    n_estimators = args['number_of_trees']  # number of trees
    max_depth = args['max_depth']  # max-depth of each tree

    # read the test data
    test_data = pd.read_csv(test_file)
    X_test = np.array(test_data.iloc[:, 0:-1])
    y_test = np.array(test_data.iloc[:, -1])

    print('RF...')

    clf = RFC(model_path, pf)

    # output
    file_num = 1
    if not os.path.exists('profile'):
        os.makedirs('profile')
    while os.path.exists(f'profile/{file_num}_proClass.txt') is True:
        file_num += 1
    file = open(f'profile/{file_num}_proClass.txt', 'w')

    m = MainProcess(clf, X_test, y_test, file, generation=20, scale=20,
                    conjunction=False, maxsat_on=True, tailor=False, fitness_func='Pro')
    param = m.pso()
    phi = param[0]
    theta = param[1]
    psi = param[2]
    k = param[3]

    ex = Extractor(clf, phi, theta, psi)
    ex.extract_forest_paths()
    ex.rule_filter()
    print(f'max_rule {ex.max_rule:>8.6f}  max_node {ex.max_node:>8.6f}')
    print("Original #rules:", ex.n_original_leaves_num)
    print('Original scale:', ex.scale)
    print("#rules after rule-filter:", len(ex._forest_values))

    sat = Z3Process(ex, k)
    sat.leaves_partition()
    sat.maxsat()
    sat.run_filter()

    print(f'#rules after MAX-SAT: {sat.n_rules_after_max}, after rule-filter: {sat.n_rules_after_filter}\n')
    print('Classes:', clf.classes_)

    f = FormulaeEstimator(sat, conjunction=True, classes=clf.classes_)
    f.get_formulae_text(file)
    print('Scale：', f.scale)
    file.close()
