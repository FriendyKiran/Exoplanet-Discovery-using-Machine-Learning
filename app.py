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

# --- Helper Functions ---
def parse_param_string(param_str):
    """Convert 'key=value, key2=value2' into dict safely."""
    parts = re.split(',\\s*(?![^()]*\\))', param_str)  # handles tuples like (50,)
    param_dict = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            param_dict[k.strip()] = v.strip()
    return param_dict

def get_best_models(all_models, dataset):
    """Return best accuracy and params for each model in a dataset."""
    best_results = []
    models_dict = all_models[dataset]

    for model_name, runs in models_dict.items():
        best_acc = -1
        best_params = None
        for param_key, entry in runs.items():
            cv_report = entry.get("cv_report", {})

            acc = 0
            # Case 1: DataFrame (classification_report output)
            if isinstance(cv_report, pd.DataFrame):
                if "accuracy" in cv_report.index and "precision" in cv_report.columns:
                    acc = cv_report.loc["accuracy", "precision"]
            # Case 2: Dict (just in case)
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

# Sidebar selection
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

# Show dropdowns for each hyperparameter
st.subheader("⚙️ Choose Hyperparameters")
selected_params = {}
for k, values in param_options.items():
    selected_params[k] = st.selectbox(k, sorted(values))

# Reconstruct key string from chosen params
param_key = ", ".join(f"{k}={v}" for k, v in selected_params.items())

# Show CV report if available
if param_key in param_sets:
    chosen_model = param_sets[param_key]["model"]
    cv_report = param_sets[param_key]["cv_report"]

    st.success("✅ Found trained model for this configuration!")
    st.write("### CV Report")
    st.dataframe(pd.DataFrame(cv_report).T)
else:
    st.warning("⚠️ This exact hyperparameter combo was not trained.")

# --- Best Models Overview ---
st.header("📊 Best Models per Dataset")

df_best = get_best_models(all_models, dataset)

# Show table
st.write("### Best Results Table")
st.dataframe(df_best)

# Show bar chart
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
