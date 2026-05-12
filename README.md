# Machine Learning Experimental Design: Random Forest Churn Prediction

## Overview
This project aims to evaluate the influence of training factors (specifically **K-fold** and **max_depth**) on the performance of the **Random Forest** algorithm in predicting customer churn.

## Project Structure
- `data/`: Contains raw and processed datasets.
  - `raw/`: Original data files (ignored by Git).
  - `processed/`: Cleaned and transformed data (ignored by Git).
- `src/`: Source code for the project (modules, utilities).
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
- `experiments/`: Scripts and configurations for running factorial experiments.
- `results/`: Output from experiments, such as model metrics, plots, and logs (ignored by Git).
- `project_blueprint/`: Documentation and project planning details.

## Setup
### 1. Initialize Virtual Environment
- **Windows**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Create a requirements.txt file with necessary libraries like scikit-learn, pandas, matplotlib, etc.)*