mk # OptExplain-Silas

Re-implementation of OptExplain ([code](https://github.com/GreeenZhang/OptExplain), [arXiv](https://arxiv.org/abs/2103.02191), [bibtex](https://dblp.org/rec/journals/corr/abs-2103-02191.html?view=bibtex)) based on [Silas](https://www.depintel.com/) models.

## Running OptExplain

1. Clone this repository and `cd` into it.

2. Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

### Preparation

1. Generate metadata using Silas.

2. Some numerical attributes in `metadata-settings.json` are treated as nominal ones in `metadata.json`. You can convert them to numerical ones by running the following command BEFORE `silas learn`:
   
    ```shell
    python modify_metadata.py metadata_file -l label_column
    ```
    
    or keep nominal features by rounding the floats AFTER `silas learn`:

    ```shell
    python modify_metadata.py metadata_file -r round_number -l label_column
    ```

   (You should modify `ROUND_NUMBER` in `OptExplain.py` to match round_number here.)
   
    This will replace metadata_file, and it should be under Silas model path before running OptExplain.

3. Ensure that the order of features in the training/testing data file matches the order of features in the metadata file.

4. If the trees do not have the key `weight` for some non-leaf node, the information gain of that node will be set to `0.1`. This can cause unexpected results.

### Running

Run `OptExplain.py` with your Silas model path, test data, and Silas prediction file on this test data:
    
```shell
python OptExplain.py -m model_path -t test_file -p prediction_file
```

NOTE: Label of test data is the last column by default. If not, `settings.json` used to learn the model should be put under model path to specify the label column.
   
Other options include

- `--generation` controls the number of PSO iterations.
- `--scale` controls the number of PSO particles.
- `--acc-weight` controls proportion of local acc in fitness computation.
- add `--conjunction` to output formulae in conjunction form.
- add `--max-sat` to apply MAX-SAT.
- add `--no-tailor` to stop using size filter.

### Outputs

OptExplain will print the results as well as save them as a file in `explanation/`.