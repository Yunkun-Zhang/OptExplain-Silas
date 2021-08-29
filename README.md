# OptExplain-Silas

Re-implementation of [OptExplain](https://arxiv.org/abs/2103.02191) ([code](https://github.com/GreeenZhang/OptExplain)) based on [Silas](https://www.depintel.com/) models.

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

    Other options include

    - `--generation` controls the number of PSO iterations.
    - `--scale` controls the number of PSO particles.
    - `--acc-weight` controls proportion of local acc in fitness computation.
    - add `--conjunction` to output formulae in conjunction form.
    - add `--max-sat` to apply MAX-SAT.
    - add `--no-tailor` to stop using size filter.

### Outputs

OptExplain will print the results as well as save them as a file in `explanation/`.
