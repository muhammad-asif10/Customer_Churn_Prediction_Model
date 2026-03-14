# Customer Churn model

Streamlit application and training workflow for predicting customer churn from
the Telco Customer Churn dataset.

## Project Overview

This repository contains:

- a training script for a logistic regression churn model
- a Streamlit app for online and batch predictions
- exploratory and model evaluation notebooks
- the dataset used to train the included model artifact

## Requirements

- Python 3.14
- Pipenv recommended for dependency management

## Setup

Install dependencies with Pipenv:

```bash
pipenv install
```

If you are not using Pipenv, install from the standard requirements file:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
pipenv run python train.py
```

This writes the trained model artifact used by the app:

- `model_C=1.0.bin`

## Run the Streamlit App

```bash
pipenv run streamlit run stream_app.py
```

By default, Streamlit serves the app locally at:

- `http://localhost:8501`

## Repository Contents

- `stream_app.py`: Streamlit user interface for churn prediction
- `train.py`: model training and serialization
- `Churn_EDA_model_development.ipynb`: exploratory analysis and preprocessing
- `Churn_model_metrics.ipynb`: evaluation metrics and model analysis
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: bundled churn dataset

## License

The source code in this repository is licensed under the Apache License 2.0.
See [LICENSE](LICENSE) for the full license text and [NOTICE](NOTICE) for
project attribution.

## Assets and Data

Code licensing does not automatically apply to every bundled asset. The dataset
and images included in this repository may be subject to separate third-party
terms. Confirm redistribution and reuse rights before using those files in
another project or distribution.

For per-file source/license status, see [ASSETS.md](ASSETS.md).

The notebook references the bundled churn dataset as coming from the Kaggle
dataset "Telco Customer Churn" by blastchar:
[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

The repository does not document the original source or license for the files
in images/, so those should be verified or replaced before wider redistribution.
