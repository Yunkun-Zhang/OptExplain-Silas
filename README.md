# OptExplain-Silas

Re-implementation of OptExplain ([code](https://github.com/GreeenZhang/OptExplain), [arXiv](https://arxiv.org/abs/2103.02191), [bibtex](https://dblp.org/rec/journals/corr/abs-2103-02191.html?view=bibtex)) based on [Silas](https://www.depintel.com/) models.

## Running OptExplain

1. Clone this repository and `cd` into it.

2. Install dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3. Run `OptExplain.py` with your Silas model path, test data, and Silas prediction file on this test data:
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

NOTE: Please ensure that the order of features in the training/testing data file matches the order of features in the metadata file.