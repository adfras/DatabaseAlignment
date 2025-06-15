# DatabaseAlignment Hint Inference Pipeline

This repository contains a demonstration pipeline to infer hint usage on EdNet interactions using hint-labeled data from the ASSISTments and KDD Cup 2010 datasets. The script `hint_inference_pipeline.py` automatically extracts the datasets, trains an XGBoost model and generates probabilities for EdNet records.

## Requirements
- Python 3.12+
- Packages: `pandas`, `scikit-learn`, `xgboost`

Install dependencies with:
```bash
pip install pandas scikit-learn xgboost
```

## Running the pipeline
Unzip the data archive and execute the script:
```bash
unzip Hint_Inference_Project_Data.zip
python3 hint_inference_pipeline.py
```
The script prints a validation report and saves `ednet_with_hints.csv` with two new columns:
- `inferred_hint_probability`
- `inferred_hint_label` (binary using 0.5 threshold)
