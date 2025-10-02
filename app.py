import streamlit as st
import joblib
import re
import pandas as pd
import altair as alt

# === Load Models ===
@st.cache_resource
def load_models():
    models = {}
    for dataset in ["keplar", "toi", "k2"]:  # adjust names to match your files
        try:
            with open(f"all_models_{dataset}.pkl", "rb") as f:
                models[dataset] = joblib.load(f)
        except FileNotFoundError:
            st.warning(f"File all_models_{dataset}.pkl not found.")
            models[dataset] = {}
    return models

all_models = load_models()

# === Load Scalers & Test Samples ===
@st.cache_resource
def load_support_files():
    try:
        scalers = joblib.load("scalers.pkl")
    except FileNotFoundError:
        st.error("❌ scalers.pkl not found!")
        scalers = {}

    try:
        test_samples = joblib.load("test_sample.pkl")
    except FileNotFoundError:
        st.error("❌ test_sample.pkl not found!")
        test_samples = {}

    return scalers, test_samples

scalers, test_samples = load_support_files()

# Label Encoding & Decoding
label_encoding = {
    "Confirmed": 1,
    "Candidate": 0,
    "False Positive": 2
}
decode_labels = {v: k for k, v in label_encoding.items()}

# --- Helper Functions ---
def parse_param_string(param_str):
    parts = re.split(',\\s*(?![^()]*\\))', param_str)
    param_dict = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            param_dict[k.strip()] = v.strip()
    return param_dict

def get_best_models(all_models, dataset):
    best_results = []
    models_dict = all_models[dataset]

    for model_name, runs in models_dict.items():
        best_acc = -1
        best_params = None
        for param_key, entry in runs.items():
            cv_report = entry.get("cv_report", {})
            acc = 0
            if isinstance(cv_report, pd.DataFrame):
                if "accuracy" in cv_report.index and "precision" in cv_report.columns:
                    acc = cv_report.loc["accuracy", "precision"]
            elif isinstance(cv_report, dict):
                acc = cv_report.get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_params = param_key
        best_results.append({
            "Model": model_name,
            "Best Accuracy": round(best_acc, 4),
            "Best Params": best_params
        })
    return pd.DataFrame(best_results)

# --- Streamlit UI ---
st.title("🚀 Model Explorer & Predictor")

tab1, tab2 = st.tabs(["📊 Model Explorer", "🔮 Prediction Playground"])

# === TAB 1: Model Explorer ===
with tab1:
    st.sidebar.header("Model Selection")
    dataset = st.sidebar.selectbox("Select dataset", list(all_models.keys()))
    model_type = st.sidebar.selectbox("Select model type", list(all_models[dataset].keys()))

    param_sets = all_models[dataset][model_type]

    # Collect all hyperparameter options
    param_options = {}
    for param_str in param_sets.keys():
        params = parse_param_string(param_str)
        for k, v in params.items():
            param_options.setdefault(k, set()).add(v)

    st.subheader("⚙️ Choose Hyperparameters")
    selected_params = {}
    for k, values in param_options.items():
        selected_params[k] = st.selectbox(k, sorted(values))

    # Reconstruct key string
    param_key = ", ".join(f"{k}={v}" for k, v in selected_params.items())

    if param_key in param_sets:
        chosen_model = param_sets[param_key]["model"]
        cv_report = param_sets[param_key]["cv_report"]
        st.success("✅ Found trained model for this configuration!")
        st.write("### CV Report")
        st.dataframe(pd.DataFrame(cv_report).T)
    else:
        st.warning("⚠️ This exact hyperparameter combo was not trained.")
        chosen_model = None

    # Best Models Overview
    st.header("📊 Best Models per Dataset")
    df_best = get_best_models(all_models, dataset)
    st.write("### Best Results Table")
    st.dataframe(df_best)
    chart = (
        alt.Chart(df_best)
        .mark_bar()
        .encode(
            x=alt.X("Model", sort="-y"),
            y="Best Accuracy",
            tooltip=["Model", "Best Accuracy", "Best Params"]
        )
    )
    st.write("### Best Accuracy Comparison")
    st.altair_chart(chart, use_container_width=True)

# === TAB 2: Prediction Playground ===
with tab2:
    st.subheader("🔮 Prediction Playground")

    dataset_choice = st.selectbox("Select Dataset for Scaler duringPrediction", list(all_models.keys()))

    st.write("### Example Input Format")
    if dataset_choice in test_samples and not test_samples[dataset_choice].empty:
        st.dataframe(test_samples[dataset_choice])
    else:
        st.info("No test sample available for this dataset.")

    uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data Preview", user_df)

        # Scale data
        if dataset_choice in scalers and scalers[dataset_choice] is not None:
          try:
            X_scaled = pd.DataFrame(scalers[dataset_choice].transform(user_df),columns=user_df.columns,index=user_df.index)
          except Exception as e:
            if isinstance(e, ValueError):
              st.error("❌ Select the proper dataset above")
            else:
              st.error(f"⚠️ Unknown error: {e}")
            X_scaled = None
        else:
          st.warning("⚠️ No scaler found for this dataset, using raw input.")
          X_scaled = user_df


        # Predictions (use model selected in Tab 1 if same dataset)
        if dataset_choice == dataset and chosen_model is not None:
          if X_scaled is not None:
            preds = chosen_model.predict(X_scaled)
            decoded_preds = [decode_labels.get(p, p) for p in preds]

            st.write("✅ Predictions")
            results = pd.DataFrame({"Prediction": decoded_preds})
            st.dataframe(results)
          else:
            st.warning("⚠️ No proper data to predict.")
        else:
            st.info(f"ℹ️ Please go to **Tab 1** and select a model for the **{dataset_choice.upper()}** dataset before predicting here.")
