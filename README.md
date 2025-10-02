# 🌌 Exoplanet Discovery using Machine Learning

This repository explores **exoplanet classification** using multiple machine learning models trained on three major space missions — **Kepler, K2, and TESS**. The aim is to build models that can distinguish between *confirmed planets, candidates, and false positives* using stellar and orbital features.  

## 🔭 Project Overview
- Implemented several models: **Random Forest, XGBoost, Gradient Boosting, LightGBM, and Multi-Layer Perceptron (MLP)**  
- Trained models **individually on Kepler, K2, and TESS datasets** to compare performance across missions  
- Stored trained models and results as `.pkl` files for reproducibility and future use  
- Evaluated models using accuracy, classification reports, and feature importance plots  

## 🚀 Interactive Streamlit App
- Built an interactive app using **Streamlit**  
- Deployed on **Streamlit Cloud** for easy access and visualization  
- Users can explore model results, visualize feature importances, and test predictions interactively  

## 📂 Repository Structure
- `all_models_*.pkl` → Saved models for each dataset  
- `all_results_*.pkl` → Evaluation metrics and results  
- `app.py` → Streamlit app code  
- `notebooks/` → Training and experimentation scripts  

## 🌠 Try It Out
👉 [Live App on Streamlit Cloud](https://exoplanet-discovery-using-machine-learning.streamlit.app/)  

---

⭐ If you find this project useful, feel free to star the repo!
