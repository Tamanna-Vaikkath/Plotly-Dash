# Homeowners Intelligence Layer (GLM + GA2M)

This project demonstrates a two-layer pricing architecture for homeowners insurance by combining a Generalized Linear Model (GLM) as the baseline and a GA2M / Explainable Boosting Machine (EBM) as an intelligence layer.

The goal is to improve pricing accuracy, capture non-linear effects and interactions, reduce adverse selection, and maintain complete transparency while keeping the overall premium risk-neutral.

---

## Key Highlights

- Improves predictive accuracy over legacy GLM using out-of-sample validation  
- Maintains risk neutrality (no increase in total premium, only redistribution)  
- Identifies underpriced and overpriced policies  
- Reclassifies policies across risk tiers  
- Reduces adverse selection  
- Provides full explainability using GA2M  

---

## Architecture Overview

Input Features → GLM Model → GA2M / EBM Intelligence Layer → Final Pure Premium

---

## Project Structure

app.py — Main Dash application  
config.py — Configuration (paths, features, constants)  
setup.py — Data generation and model training  

data/final_predictions.csv — Dataset with predictions  

models/frequency_glm.pkl — Frequency model  
models/severity_glm.pkl — Severity model  
models/ebm_model.pkl — GA2M model  
models/preprocessor.pkl — GLM preprocessing pipeline  

assets/vm_logo.png — Logo file  

---

## Core Concepts

### GLM (Baseline Layer)
Linear, interpretable model capturing primary risk signals but limited in handling non-linearity and interactions.

### GA2M / EBM (Intelligence Layer)
Captures:
- Non-linear relationships  
- Feature interactions  

Acts as a residual correction layer on top of GLM while remaining fully explainable.

### Risk Neutrality
Ensures total premium remains unchanged:
Average uplift = 1 → redistribution only, no inflation.

---

## Dashboard Overview

### Tab 1: Business Case
- Premium redistribution  
- Reclassification  
- Adverse selection correction  
- Risk neutrality validation  

### Tab 2: Intelligence Signals
- Shape functions  
- Feature importance  
- Interaction discovery  
- H-statistic validation  

### Tab 3: Policy Lens
- Policy-level analysis  
- GLM vs Final comparison  
- Waterfall explanation  

### Tab 4: Framework
- Model architecture  
- Feature tiers  
- Pricing logic  

---

## Key Metrics

- R² improvement  
- RMSE comparison  
- % policies repriced (>10%)  
- % underpriced (before vs after)  
- Premium movement  
- Risk neutrality check  

---

## Installation

pip install -r requirements.txt  

python setup.py  

python app.py  

---

## Tech Stack

- Python  
- Plotly Dash  
- Bootstrap  
- GLM (statsmodels / sklearn)  
- EBM (InterpretML)  

---

## Business Value

- Explainable AI (glass-box model)  
- Reduced adverse selection  
- Improved pricing accuracy  
- Better risk segmentation  
- Revenue-neutral optimization  
- Actionable underwriting insights  

---

## Author

Tamanna Vaikkath
