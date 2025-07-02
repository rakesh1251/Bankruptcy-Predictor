# 🏦 Bankruptcy Predictor with SHAP Explainability

This project is an advanced financial risk prediction tool that uses machine learning to estimate the probability of bankruptcy based on 60+ financial ratios. It features an interactive Streamlit web app with SHAP-based model explainability.

**Live Demo:** [Hugging Face App](https://huggingface.co/spaces/ranjan51/bankruptcy_predictor)

---

## Features

- Predict bankruptcy risk for 1 to 5 years ahead  
- Input 30+ financial ratios using sliders  
- Uses a tuned XGBoost classification model  
- SHAP waterfall plots to explain each prediction  
- Simplified UI for decision-makers and analysts  

---

## Project Structure
* Bankruptcy-Predictor/
- app.py (Streamlit app with inputs and SHAP integration)
- xgb_tuned_bst.joblib (Pre-trained XGBoost model)
- requirements.txt (Python dependencies)
- README.md (Project documentation)


---

## Model Details

- **Model**: XGBoost Classifier (tuned via GridSearchCV)  
- **Training Data**: Bankruptcy financial ratio dataset (64 features)  
- **Handling Imbalance**: RandomOverSampler  
- **Forecasting**: One-hot encoded forecast window (1–5 years ahead)  

---

## How to Run Locally

```bash
git clone https://github.com/rakesh1251/Bankruptcy-Predictor.git
cd Bankruptcy-Predictor
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
- This project is for educational and demonstration purposes only.
- It should not be used for real-world financial decisions.

## Author
Ranjan Rakesh
🔗 [LinkedIn](https://www.linkedin.com/in/rakesh-mallik/)
🚀 [Live App on Hugging Face](https://huggingface.co/spaces/ranjan51/bankruptcy_predictor)
