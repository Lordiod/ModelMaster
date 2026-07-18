"""
ModelMaster - Streamlit Application Entry Point

Deploy with: streamlit run streamlit_app.py
"""

import matplotlib
matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.controller.model_controller import ModelController

# --------------------------------------------------------------------------- #
# Page configuration & styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="ModelMaster",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .mm-hero {
        padding: 1.75rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .mm-hero h1 {margin: 0; font-size: 1.9rem; font-weight: 700;}
    .mm-hero p {margin: .35rem 0 0 0; opacity: .92; font-size: 1rem;}

    .mm-step-label {
        display: inline-block;
        background: #16a34a;
        color: #fff;
        border-radius: 999px;
        width: 1.6rem;
        height: 1.6rem;
        text-align: center;
        line-height: 1.6rem;
        font-size: .85rem;
        font-weight: 700;
        margin-right: .5rem;
    }
    .mm-section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: .25rem;
    }
    .mm-card {
        border: 1px solid rgba(120,120,120,.25);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        background: rgba(120,120,120,.04);
    }
    .mm-metric-grid div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    div[data-testid="stStatusWidget"] {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

ALGORITHMS = {
    "Decision Tree": "tree",
    "Linear Regression": "linear_reg",
    "Logistic Regression": "logistic_reg",
    "K-Nearest Neighbors (KNN)": "knn",
    "Support Vector Machine (SVM)": "svm",
}

SAMPLE_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
SAMPLE_DATASET_LABELS = {
    "iris.csv": "🌸 Iris Flower Classification",
    "diabetes.csv": "🩺 Diabetes Prediction",
    "pima-indians-diabetes.csv": "🩺 Pima Indians Diabetes",
    "Titanic.csv": "🚢 Titanic Survival",
    "fake_bills.csv": "💵 Counterfeit Bill Detection",
    "lung cancer survey.csv": "🫁 Lung Cancer Survey",
}


def list_sample_datasets():
    """Returns the CSV filenames available in the bundled Datasets folder, sorted by label."""
    if not os.path.isdir(SAMPLE_DATASETS_DIR):
        return []
    files = [f for f in os.listdir(SAMPLE_DATASETS_DIR) if f.lower().endswith(".csv")]
    return sorted(files, key=lambda f: SAMPLE_DATASET_LABELS.get(f, f))

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "controller" not in st.session_state:
    st.session_state.controller = ModelController()
if "loader_controller" not in st.session_state:
    st.session_state.loader_controller = ModelController()
if "loaded_dataset_name" not in st.session_state:
    st.session_state.loaded_dataset_name = None
if "loaded_model_name" not in st.session_state:
    st.session_state.loaded_model_name = None
if "trained_algorithm_label" not in st.session_state:
    st.session_state.trained_algorithm_label = None


def reset_session():
    """Resets the training workspace to a clean state."""
    st.session_state.controller = ModelController()
    st.session_state.loaded_dataset_name = None
    st.session_state.trained_algorithm_label = None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## 🧠 ModelMaster")
    st.caption("Advanced Learning Laboratory for Intelligent Networks")
    st.divider()
    st.markdown(
        "**Workflow**\n"
        "1. Upload a CSV or pick a sample dataset\n"
        "2. Choose feature count (RFE)\n"
        "3. Pick an algorithm & tune it\n"
        "4. Train, test, and review results\n"
        "5. Download your trained model"
    )
    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        reset_session()
        st.rerun()
    st.divider()
    with st.expander("About ModelMaster"):
        st.write(
            "ModelMaster is a comprehensive tool for machine learning model "
            "development and evaluation, featuring multiple classification "
            "algorithms including SVM, KNN, Decision Trees, and Logistic "
            "Regression.\n\n"
            "**Key Features**\n"
            "- Intuitive dataset loading and preprocessing\n"
            "- Recursive Feature Elimination (RFE) for feature ranking\n"
            "- Customizable model parameters and configurations\n"
            "- Detailed performance metrics and visualization tools\n"
            "- Model persistence for saving and loading trained models\n\n"
            "Version 2.0.0 · Streamlit Edition"
        )

st.markdown(
    """
    <div class="mm-hero">
        <h1>ModelMaster</h1>
        <p>Train, evaluate, and export classification models without writing a line of code.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_train, tab_load = st.tabs(["🚀 Train a New Model", "📂 Load an Existing Model"])

# --------------------------------------------------------------------------- #
# Tab 1 — Train a new model
# --------------------------------------------------------------------------- #

with tab_train:
    controller: ModelController = st.session_state.controller

    st.markdown('<div class="mm-section-title"><span class="mm-step-label">1</span>Load Your Dataset</div>', unsafe_allow_html=True)

    dataset_source = st.radio(
        "Dataset source",
        ["Upload my own CSV", "Use a sample dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if dataset_source == "Upload my own CSV":
        uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"], key="dataset_uploader")

        if uploaded_file is not None and st.session_state.loaded_dataset_name != uploaded_file.name:
            result = controller.load_dataset(uploaded_file)
            if result["success"]:
                st.session_state.loaded_dataset_name = uploaded_file.name
                st.session_state.trained_algorithm_label = None
            else:
                st.session_state.loaded_dataset_name = None
                st.error(result["message"])
    else:
        sample_files = list_sample_datasets()
        if not sample_files:
            st.warning("No sample datasets were found in the `Datasets/` folder.")
        else:
            selected_sample = st.selectbox(
                "Choose a sample dataset",
                sample_files,
                format_func=lambda f: SAMPLE_DATASET_LABELS.get(f, f),
                key="sample_dataset_select",
            )
            sample_id = f"sample::{selected_sample}"
            if st.session_state.loaded_dataset_name != sample_id:
                result = controller.load_dataset(os.path.join(SAMPLE_DATASETS_DIR, selected_sample))
                if result["success"]:
                    st.session_state.loaded_dataset_name = sample_id
                    st.session_state.trained_algorithm_label = None
                else:
                    st.session_state.loaded_dataset_name = None
                    st.error(result["message"])

    dataset_ready = (
        controller.current_model is not None
        and controller.current_model.dataset is not None
    )

    if dataset_ready:
        dataset = controller.current_model.dataset
        max_features = dataset.shape[1] - 1

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", f"{dataset.shape[0]:,}")
        col_b.metric("Columns", f"{dataset.shape[1]:,}")
        col_c.metric("Available Features", f"{max_features:,}")

        with st.expander("Preview dataset", expanded=False):
            st.dataframe(dataset.head(10), use_container_width=True)

        st.markdown('<div class="mm-section-title"><span class="mm-step-label">2</span>Select Feature Count</div>', unsafe_allow_html=True)
        st.caption("Recursive Feature Elimination (RFE) will select the top N features for training.")
        num_features = st.number_input(
            "Number of features to use",
            min_value=1,
            max_value=max_features,
            value=min(5, max_features),
            step=1,
        )

        st.markdown('<div class="mm-section-title"><span class="mm-step-label">3</span>Choose an Algorithm</div>', unsafe_allow_html=True)
        algo_label = st.selectbox("Algorithm", list(ALGORITHMS.keys()))
        algorithm_code = ALGORITHMS[algo_label]

        kwargs = {}
        if algorithm_code == "svm":
            kwargs["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        elif algorithm_code == "knn":
            kwargs["n_neighbors"] = st.number_input("Number of neighbors", min_value=1, value=5, step=1)
        elif algorithm_code == "tree":
            kwargs["max_depth"] = st.number_input("Max depth", min_value=1, value=5, step=1)

        st.markdown('<div class="mm-section-title"><span class="mm-step-label">4</span>Train &amp; Test</div>', unsafe_allow_html=True)
        test_size = st.slider("Test split ratio", min_value=0.05, max_value=0.95, value=0.2, step=0.05)

        train_col, test_col = st.columns(2)
        with train_col:
            if st.button("▶️ Train Model", use_container_width=True, type="primary"):
                controller.set_feature_count(num_features)
                controller.select_algorithm(algorithm_code)
                result = controller.train_model(algorithm_code, test_size, **kwargs)
                if result["success"]:
                    st.session_state.trained_algorithm_label = algo_label
                    st.success(result["message"])
                else:
                    st.session_state.trained_algorithm_label = None
                    st.error(result["message"])
        with test_col:
            # Computed after the Train button so a fresh train in this same
            # run immediately re-enables Test, without needing a rerun that
            # would wipe the success/error message above before it renders.
            can_test = controller.current_model.model is not None
            if st.button("✅ Test Model", use_container_width=True, disabled=not can_test):
                result = controller.test_model()
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])

        metrics = controller.get_metrics()
        if metrics and metrics["accuracy"] is not None:
            st.markdown('<div class="mm-section-title"><span class="mm-step-label">5</span>Results</div>', unsafe_allow_html=True)
            if st.session_state.trained_algorithm_label:
                st.caption(f"Showing results for: **{st.session_state.trained_algorithm_label}**")

            with st.container():
                st.markdown('<div class="mm-card mm-metric-grid">', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                m2.metric("Precision", f"{metrics['precision']:.4f}")
                m3.metric("Recall", f"{metrics['recall']:.4f}")
                m4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

            fig_col, download_col = st.columns([2, 1])
            with fig_col:
                fig = controller.get_confusion_matrix_figure()
                if fig is not None:
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
            with download_col:
                model_bytes = controller.get_model_bytes()
                if model_bytes:
                    st.download_button(
                        "💾 Download Trained Model (.pkl)",
                        data=model_bytes,
                        file_name=f"{algorithm_code}_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
    else:
        st.info("Upload a CSV dataset above to get started.")

# --------------------------------------------------------------------------- #
# Tab 2 — Load an existing model
# --------------------------------------------------------------------------- #

with tab_load:
    loader: ModelController = st.session_state.loader_controller

    st.markdown('<div class="mm-section-title">Load a Previously Saved Model</div>', unsafe_allow_html=True)
    st.caption("Upload a `.pkl` file produced by ModelMaster to inspect it and run predictions on new data.")

    uploaded_model = st.file_uploader("Upload a model file", type=["pkl"], key="model_uploader")

    if uploaded_model is not None and st.session_state.loaded_model_name != uploaded_model.name:
        result = loader.load_model(uploaded_model)
        st.session_state.loaded_model_name = uploaded_model.name
        if result["success"]:
            st.session_state.loaded_model_info = result["model_info"]
            st.success(result["message"])
        else:
            st.session_state.loaded_model_info = None
            st.error(result["message"])

    model_info = st.session_state.get("loaded_model_info")

    if model_info:
        n_features = loader.get_model_feature_count()
        details_tab, predict_tab = st.tabs(["🔎 Model Details", "🎯 Predict on New Data"])

        with details_tab:
            st.markdown('<div class="mm-card">', unsafe_allow_html=True)
            st.markdown(f"**Algorithm Type:** `{model_info['class_name']}`")
            for key, value in model_info["parameters"].items():
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            if n_features is not None:
                st.markdown(f"**Expected Input Features:** {n_features}")
            st.markdown('</div>', unsafe_allow_html=True)

        with predict_tab:
            if n_features is not None:
                st.caption(
                    f"Upload a CSV with **{n_features} feature column(s)**, in the same order "
                    "used during training and with no target column."
                )
            else:
                st.caption(
                    "Upload a CSV containing only the feature columns used during training "
                    "(no target column)."
                )

            predict_file = st.file_uploader("Upload feature data (CSV)", type=["csv"], key="predict_uploader")

            if predict_file is not None:
                try:
                    predict_df = pd.read_csv(predict_file)
                except Exception as e:
                    predict_df = None
                    st.error(f"Could not read CSV: {e}")

                if predict_df is not None:
                    st.dataframe(predict_df.head(10), use_container_width=True)

                    exclude_cols = []
                    if n_features is not None and predict_df.shape[1] > n_features:
                        st.info(
                            f"This file has {predict_df.shape[1]} column(s), but the model "
                            f"expects {n_features}. If it still includes the target/label "
                            "column, select it below to exclude it before predicting."
                        )
                        default_exclude = (
                            [predict_df.columns[-1]]
                            if predict_df.shape[1] == n_features + 1
                            else []
                        )
                        exclude_cols = st.multiselect(
                            "Column(s) to exclude",
                            options=list(predict_df.columns),
                            default=default_exclude,
                        )

                    feature_df = predict_df.drop(columns=exclude_cols) if exclude_cols else predict_df

                    if n_features is not None and feature_df.shape[1] != n_features:
                        st.warning(
                            f"After exclusions this file has {feature_df.shape[1]} column(s), "
                            f"but the model expects {n_features}. Predictions may fail or be inaccurate."
                        )

                    if st.button("🎯 Run Predictions", type="primary"):
                        result = loader.predict(feature_df.values)
                        if result["success"]:
                            result_df = predict_df.copy()
                            result_df["Prediction"] = result["predictions"]
                            st.success("Predictions generated successfully!")
                            st.dataframe(result_df, use_container_width=True)
                            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "💾 Download Predictions (.csv)",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(result["message"])
            else:
                st.info("Upload a CSV file with feature columns to generate predictions.")
    else:
        st.info("Upload a `.pkl` model file above to view its details and make predictions.")
