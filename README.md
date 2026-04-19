**Initial Production-Level Project Structure**

```
liquidity-stress-early-warning/
│
├── .github/                    # CI/CD workflows 
│
├── configs/                   # Configuration files (YAML)
│   ├── config.yaml
│   └── model_params.yaml
│
├── data/
│   ├── raw/                   # Original immutable data
│   ├── interim/               # Cleaned but not final
│   └── processed/             # Final modeling data
│
├── notebooks/                 # experimental notebooks 
│   └── eda.ipynb
│
├── src/                       # Core source code 
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   └── helpers.py
│
├── tests/                     # Unit tests 
│
├── artifacts/                 # Saved models, transformers
│
├── requirements.txt
├── setup.py                   # Makes project installable
├── .gitignore
└── README.md
```
