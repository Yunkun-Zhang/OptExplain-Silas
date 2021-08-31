import os
import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from silas import RFC
from Main_Process import MainProcess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', default='model/flights/',
                        help='root of your Silas model')
    parser.add_argument('-t', '--test-file', default='tests/clean-flights_test.csv',
                        help='path to your test file')
    parser.add_argument('-p', '--prediction-file', default='tests/predictions_flights.csv',
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

    generation = args['generation']
    scale = args['scale']
    acc_weight = args['acc_weight']
    conjunction = args['conjunction']
    maxsat_on = args['max_sat']
    size_filter = not args['no_tailor']

    # read the test data and adjust data type
    with open(os.path.join(model_path, 'settings.json')) as f:
        settings = json.load(f)
    with open(os.path.join(model_path, 'metadata.json')) as f:
        metadata = json.load(f)
    label = settings['output-feature']
    test_data = pd.read_csv(test_file)
    columns = list(test_data.columns)
    label_column = columns.index(label)
    test_data = test_data.values.tolist()
    for i, f in enumerate(metadata['attributes']):
        if f['type'] == 'nominal':
            for sample in range(len(test_data)):
                test_data[sample][i] = str(test_data[sample][i])
    X_test = [sample[:label_column] + sample[label_column + 1:] for sample in test_data]
    y_test = [sample[label_column] for sample in test_data]

    # create random forest from Silas
    print('RF...', end='\r')
    clf = RFC(model_path, pf)
    print('RF acc:', accuracy_score(y_test, clf.predict()))

    # output
    file_num = 1
    if not os.path.exists('explanation'):
        os.makedirs('explanation')
    while os.path.exists(f'explanation/{file_num}.txt') is True:
        file_num += 1
    file = open(f'explanation/{file_num}.txt', 'w')
    file.write('generation = {}\tscale = {}\tacc_weight = {}\tmaxsat = {}\ttailor = {}\n'.
               format(generation, scale, acc_weight, maxsat_on, size_filter))
    print('explain...')
    file.write('begin\n')
    m = MainProcess(clf, X_test, y_test, file, generation=generation, scale=scale, acc_weight=acc_weight,
                    conjunction=conjunction, maxsat_on=maxsat_on, tailor=size_filter, fitness_func='Opt')
    best_param = m.pso()
    m.explain(best_param, auc_plot=False)
    file.close()
