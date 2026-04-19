**Initial Production-Level Project Structure**

```
liquidity-stress-early-warning/
│
├── .github/                    # CI/CD workflows (later phase)
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
├── notebooks/                 # EDA notebooks (NOT production code)
│   └── eda.ipynb
│
├── src/                       # Core source code (THIS IS THE HEART)
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
├── tests/                     # Unit tests (optional but senior-level)
│
├── artifacts/                 # Saved models, transformers
│
├── requirements.txt
├── setup.py                   # Makes project installable
├── .gitignore
└── README.md
```
