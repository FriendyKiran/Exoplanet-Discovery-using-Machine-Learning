import streamlit as st
import joblib
import re
import pandas as pd
import altair as alt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import time


st.set_page_config(
    page_title="ExoAI: Interactive Machine Learning for Exoplanet Discovery",
    page_icon="🚀",
    layout="wide"
)

data_dict = {'TESS':'toi', 'K2':'k2', 'Keplar':'keplar'}
# === Load Models ===
@st.cache_resource
def load_models():
    models = {}
    for dataset in ["Keplar", "TESS", "K2"]:  # adjust names to match your files
        try:
            with open(f"all_models_{data_dict[dataset]}.pkl", "rb") as f:
                models[dataset] = joblib.load(f)
        except FileNotFoundError:
            st.warning(f"File all_models_{data_dict[dataset]}.pkl not found.")
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

# === Load Test Summary ===
@st.cache_resource
def load_test_summary():
    try:
        return joblib.load("test_summary.pkl")
    except FileNotFoundError:
        st.error("❌ test_summary.pkl not found!")
        return {}

test_summary = load_test_summary()

# === Load Data ===
@st.cache_resource
def load_train_test_data():
    try:
        train_data = joblib.load("train_data.pkl")
    except FileNotFoundError:
        st.error("❌ train_data.pkl not found!")
        train_data = {}

    try:
        test_data = joblib.load("test_data.pkl")
    except FileNotFoundError:
        st.error("❌ test_data.pkl not found!")
        test_data = {}

    return train_data, test_data

train_data, test_data = load_train_test_data()


import base64

# === Set Streamlit Background ===
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/webp;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    # [data-testid="stSidebar"] {{
    #     background-color: rgba(10, 10, 10, 0.85);
    # }}

    # [data-testid="stHeader"] {{
    #     background: rgba(0, 0, 0, 0.5);
    # }}

    # h1, h2, h3, h4, h5, h6, p, span, div {{
    #     color: #ffffff !important;
    # }}

    # .stButton > button {{
    #     color: black !important;
    #     background-color: rgba(255, 255, 255, 0.85);
    #     border-radius: 10px;
    #     border: 1px solid #333;
    # }}
    # .stButton > button:hover {{
    #     background-color: rgba(240, 240, 240, 0.9);
    #     border: 1px solid #000;
    # }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function early in your app
set_background("background.jpg")

st.markdown("""
    <style>
    /* === Custom Dark Radio Section Styling === */
    div[data-testid="stRadio"] {
        background: rgba(20, 20, 30, 0.85);
        padding: 1.2em 1.5em;
        border-radius: 12px;
        border: 1px solid rgba(0, 170, 255, 0.3);
        box-shadow: 0 0 12px rgba(0, 170, 255, 0.2);
        color: #e0e0e0;
        transition: 0.3s ease;
    }
    div[data-testid="stRadio"]:hover {
        box-shadow: 0 0 20px rgba(0, 170, 255, 0.4);
    }
    div[data-testid="stRadio"] label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    /* ====== Custom Dark Selectbox Styling ====== */
    div[data-baseweb="select"] > div {
        background-color: #262627 !important;
        border: 1px solid rgba(0, 170, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        transition: 0.3s ease;
    }
    div[data-baseweb="select"]:hover > div {
        box-shadow: 0 0 10px rgba(0, 170, 255, 0.4) !important;
    }
    div[data-baseweb="select"] span {
        color: #e0e0e0 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #00aaff !important;
    }
    /* Dropdown menu background */
    ul[role="listbox"] {
        background-color: #262627 !important;
        border: 1px solid rgba(0, 170, 255, 0.3) !important;
    }
    ul[role="listbox"] li:hover {
        background-color: rgba(0, 170, 255, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
st.title("🪐 ExoAI: Intelligent Exploration of New Worlds")

tab1, tab2, tab3 = st.tabs(["📊 Model Explorer", "🔮 Prediction Playground", "🧠 Build Your Own Model"])

# === TAB 1: Model Explorer ===
with tab1:
    st.sidebar.header("Model Selection")
    dataset = st.sidebar.selectbox("📀 Select dataset", list(all_models.keys()))
    model_type = st.sidebar.selectbox("🧠 Select model type", list(all_models[dataset].keys()))

    param_sets = all_models[dataset][model_type]

    # Collect all hyperparameter options
    param_options = {}
    for param_str in param_sets.keys():
        params = parse_param_string(param_str)
        for k, v in params.items():
            param_options.setdefault(k, set()).add(v)

    st.sidebar.subheader("⚙️ Choose Hyperparameters")
    selected_params = {}
    for k, values in param_options.items():
        selected_params[k] = st.sidebar.selectbox(k, sorted(values))

    # Reconstruct key string
    param_key = ", ".join(f"{k}={v}" for k, v in selected_params.items())

    st.subheader(f"📊 {model_type} Model Overview ({dataset} Dataset)")
    if param_key in param_sets:
        chosen_model = param_sets[param_key]["model"]
        cv_report = param_sets[param_key]["cv_report"]
        st.success("✅ Found trained model for this configuration!")
        st.write("### CV Report")
        st.dataframe(pd.DataFrame(cv_report).T)
    else:
        st.warning("⚠️ This exact hyperparameter combo was not trained.")
        chosen_model = None

    # Best Models Overview (Final Test Results)
    st.subheader(f"📊 Best Models Performance on {dataset} Test Set")

    if dataset in test_summary:
        df_test = test_summary[dataset].sort_values("test_accuracy", ascending=False)

        # st.write("### Sorted by Test Accuracy")
        st.dataframe(df_test)

        # Plot using test_accuracy instead of CV
        bars = (
            alt.Chart(df_test, height=500)
            .mark_bar(size=40)
            .encode(
                x=alt.X(
                    "model_name",
                    sort="-y",
                    axis=alt.Axis(labelAngle=0, title="Model")
                ),
                y=alt.Y(
                    "test_accuracy",
                    title="Test Accuracy"
                ),
                tooltip=["model_name", "cv_accuracy", "test_accuracy", "best_params"]
            )
        )

        text = (
            alt.Chart(df_test, height=500)
            .mark_text(
                align="center",
                baseline="bottom",
                dy=-5  # move text a bit above the bar
            )
            .encode(
                x="model_name",
                y="test_accuracy",
                text=alt.Text("test_accuracy:Q", format=".3f")  # 3 decimal places
            )
        )

        chart = bars + text
        st.write("### Test Accuracy Comparison")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("⚠️ No test summary available for this dataset.")


# === TAB 2: Prediction Playground ===
with tab2:
    st.subheader("🔮 Prediction Playground")

    dataset_choice = st.selectbox("Select Dataset for Scaler duringPrediction", list(all_models.keys()))

    st.write("### Example Input Format")
    if dataset_choice in test_samples and not test_samples[dataset_choice].empty:
        st.dataframe(test_samples[dataset_choice])
    else:
        st.info("No test sample available for this dataset.")

    # === Step 1: Choose Input Source ===
    st.markdown("### 🧩 Step 1 – Choose Input Data Source")
    data_option = st.radio(
        "Would you like to upload your own data or use a provided test dataset?",
        ("Upload my own CSV", "Use default NASA test dataset"),
        key=f"data_option_{dataset_choice}"
    )

    user_df = None

    # === If user uploads their own CSV ===
    if data_option == "Upload my own CSV":
        uploaded_file = st.file_uploader("📤 Upload CSV file with features", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.write("📂 Uploaded Data Preview")
                st.dataframe(user_df)
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")

    # === If user chooses default dataset ===
    elif data_option == "Use default NASA test dataset":
        if dataset_choice in test_data:
            X_test_default, y_test_default = test_data[dataset_choice]

            # ✅ Use only top 10 rows for quick prediction demo
            X_test_default = X_test_default.head(50)
            if y_test_default is not None and hasattr(y_test_default, "head"):
                y_test_default = y_test_default.head(50)

            st.success(f"✅ Loaded default test dataset for {dataset_choice} (showing top 10 rows)")
            st.dataframe(X_test_default)
            user_df = X_test_default
        else:
            st.warning(f"⚠️ No default test dataset found for {dataset_choice}")


    if user_df is not None:

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
            st.info(f"ℹ️ Please go to **Side panel** and select a model for the **{dataset_choice.upper()}** dataset before predicting here.")

# === TAB 3: Train Your Own Model (Progressive Flow with conditional rendering) ===

with tab3:
    st.subheader("🧠 Train Your Own Model")
    st.markdown("Follow each step sequentially to configure and train your own exoplanet classifier.")

    # === Model Base Constructors ===
    models = {
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "LightGBM": LGBMClassifier,
        "MLP": MLPClassifier,
    }

    # === Step 1: Choose Model ===
    st.markdown("### 🧭 Step 1 – Choose a Model")
    model_name = st.selectbox("Select Model", [""] + list(models.keys()), key="model_select")

    if model_name:
        st.success(f"✅ Model '{model_name}' selected. Configure hyperparameters below.")

        # === Step 2: Dynamic Hyperparameter Controls ===
        st.markdown("### ⚙️ Step 2 – Set Hyperparameters")
        user_params = {}

        if model_name == "RandomForest":
            n_est = st.slider("n_estimators", 100, 1000, 200, 10, key=f"{model_name}_nest")
            max_depth = st.selectbox("max_depth", ["None"] + list(range(2, 31)), key=f"{model_name}_depth")
            min_split = st.slider("min_samples_split", 2, 10, 2, key=f"{model_name}_split")
            user_params = {
                "n_estimators": n_est,
                "max_depth": max_depth,
                "min_samples_split": min_split,
                "random_state": 42,
            }

        elif model_name == "XGBoost":
            n_est = st.slider("n_estimators", 100, 1000, 200, 10, key=f"{model_name}_nest")
            max_depth = st.slider("max_depth", 2, 15, 5, key=f"{model_name}_depth")
            lr = st.slider("learning_rate", 0.001, 0.3, 0.1, 0.01, key=f"{model_name}_lr")
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key=f"{model_name}_subs")
            user_params = {
                "n_estimators": n_est,
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": subsample,
                "eval_metric": "mlogloss",
                "random_state": 42,
            }

        elif model_name == "GradientBoosting":
            n_est = st.slider("n_estimators", 100, 1000, 200, 10, key=f"{model_name}_nest")
            lr = st.slider("learning_rate", 0.001, 0.3, 0.1, 0.01, key=f"{model_name}_lr")
            max_depth = st.slider("max_depth", 2, 15, 5, key=f"{model_name}_depth")
            user_params = {
                "n_estimators": n_est,
                "learning_rate": lr,
                "max_depth": max_depth,
                "random_state": 42,
            }

        elif model_name == "LightGBM":
            n_est = st.slider("n_estimators", 100, 1000, 200, 100, key=f"{model_name}_nest")
            num_leaves = st.slider("num_leaves", 16, 128, 31, 8, key=f"{model_name}_leaves")
            lr = st.slider("learning_rate", 0.001, 0.3, 0.1, 0.01, key=f"{model_name}_lr")
            user_params = {
                "n_estimators": n_est,
                "num_leaves": num_leaves,
                "learning_rate": lr,
                "verbose": -1,
                "random_state": 42,
            }

        elif model_name == "MLP":
            hidden = st.selectbox("hidden_layer_sizes", ["(50,)", "(100,)", "(50,50)", "(100,50)"], key=f"{model_name}_hidden")
            act = st.selectbox("activation", ["relu", "tanh", "logistic"], key=f"{model_name}_act")
            lr = st.slider("learning_rate_init", 0.0001, 0.1, 0.001, 0.001, key=f"{model_name}_lr")
            user_params = {
                "hidden_layer_sizes": eval(hidden),
                "activation": act,
                "learning_rate_init": lr,
                "max_iter": 1000,
                "random_state": 42,
            }

        st.success("✅ Model configuration complete. Please select the dataset to train.")

        # === Step 3: Select Dataset ===
        st.markdown("### 🧩 Step 3 – Select Dataset")
        dataset_choice = st.selectbox(
            "Choose Dataset to Train On",
            [""] + list(train_data.keys()) if train_data else [],
            key="dataset_select",
        )

        if dataset_choice:
            st.success(f"✅ Dataset '{dataset_choice}' selected. These are scaled feature values used for training.")
            

            X_train, y_train = train_data[dataset_choice]
            X_test, y_test = test_data.get(dataset_choice, (None, None))
            st.write(f"📊 **Preview of {dataset_choice} Training Data**")
            st.dataframe(X_train.head())

            # === Step 4: Train Button ===
            st.markdown("### 🚀 Step 4 – Train the Model")

            # A unique key per (model,dataset)
            state_key = f"{model_name}_{dataset_choice}_trained"

            if state_key not in st.session_state:
                st.session_state[state_key] = False

            if not st.session_state[state_key]:
                if st.button("Start Training", key=f"train_{model_name}_{dataset_choice}"):
                    with st.spinner("🌠 Launching interstellar AI probe... Hold tight!"):
                        start_time = time.time()

                        # Countdown
                        for i in range(3, 0, -1):
                            st.markdown(f"## ⏳ Launching in T-{i}...")
                            time.sleep(1)

                        st.markdown("## 🧠 Training Model in Deep Space...")
                        progress = st.progress(0)
                        phases = [
                            "Initializing data engines...",
                            "Activating neural warp drive...",
                            "Collecting cosmic gradients...",
                            "Docking with convergence orbit..."
                        ]

                        current_progress = 0
                        for phase in phases[0:3]:
                            st.info(phase)
                            for _ in range(30):
                                current_progress += 1
                                time.sleep(0.04)
                                progress.progress(current_progress / 100.0)

                        # Train
                        params_fixed = {k: (None if v == "None" else v) for k, v in user_params.items()}
                        model = models[model_name](**params_fixed)
                        model.fit(X_train, y_train)
                        st.info(phases[3])
                        for _ in range(10):
                            current_progress += 1
                            time.sleep(0.04)
                            progress.progress(current_progress / 100.0)

                        elapsed = time.time() - start_time
                        st.session_state[state_key] = True
                        st.session_state["trained_model"] = model
                        st.success(f"✅ Training complete for {model_name} on {dataset_choice}! ⏱ {elapsed:.2f}s")
                        st.balloons()

            # === Step 5: Evaluation & Download ===
            if st.session_state.get(state_key, False):
                model = st.session_state["trained_model"]

                if X_test is not None:
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.markdown(f"### 🌟 Overall Accuracy: **{acc:.4f}**")

                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    st.write("### 📈 Detailed Classification Report on Test dataset")
                    st.dataframe(pd.DataFrame(report).T)
                else:
                    st.warning("⚠️ No test data found for this dataset.")

                # === Download only on click ===
                import io
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)
                st.download_button(
                    label="💾 Download Trained Model",
                    data=buffer,
                    file_name=f"{model_name}_{dataset_choice}_trained.pkl",
                    mime="application/octet-stream",
                    key=f"download_{model_name}_{dataset_choice}",
                )
