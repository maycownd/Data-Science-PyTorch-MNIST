Data Science Template
==============================

Template for the project development.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- Source code for use in this project.
        │
        ├── deployment           <- Scripts to deploy the model
        │   └── deploy.py
        │
        ├── preprocessing       <- Scripts to prepare the data for modeling
        │   └── preprocess.py
        │
        ├── training         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   └── train.py
        │
        └── tests            <- Scripts to create tests to check quality of the code
            |── test_functions.py
            |
            └── unit_test.py


--------
