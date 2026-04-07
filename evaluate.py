import os
import time
import joblib
import shap
import mlflow
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
import config

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)

def setup_plots():
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    })

def ensure_dirs():
    os.makedirs(config.VIZ_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(config.VIZ_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def evaluate():
    setup_plots()
    ensure_dirs()

    while not os.path.exists(config.MODEL_PATH):
        print(f"Waiting for model {config.MODEL_PATH}...")
        time.sleep(5)

    df = config.load_data()

    drop_cols = [config.TARGET]
    if not config.WITH_GENDER:
        drop_cols.append("Sex")

    X = df.drop(columns=drop_cols)
    y = config.encode_target(df[config.TARGET])

    sensitive = df[config.PROTECTED_ATTR] if config.WITH_GENDER else None

    if config.WITH_GENDER:
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive,
            test_size=0.2,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )

    pipeline = joblib.load(config.MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bad", "Good"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    save_fig("confusion_matrix")

    # Performance Metrics
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_Macro": f1_score(y_test, y_pred, average="macro"),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC_AUC": roc_auc,
    }

    # Log metrics to MLflow
    mlflow.set_experiment("Credit_Risk_Assessment")
    with mlflow.start_run():
        mlflow.log_metrics(metrics)

    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    save_fig("metrics_bar")

    # Fairness Analysis
    if config.WITH_GENDER:
        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=s_test,
        )

        dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
        eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)

        fairness_results = {
            "Demographic Parity Diff": dp_diff,
            "Equalized Odds Diff": eo_diff,
            "Selection Rate Diff": mf.by_group["selection_rate"].max() - mf.by_group["selection_rate"].min(),
            "FPR Diff": mf.by_group["false_positive_rate"].max() - mf.by_group["false_positive_rate"].min(),
            "FNR Diff": mf.by_group["false_negative_rate"].max() - mf.by_group["false_negative_rate"].min(),
        }

        mf.by_group.plot(kind="bar")
        plt.ylabel("Metric Value")
        plt.title("Group-wise Fairness Metrics")
        save_fig("fairness_group_metrics")

        plt.bar(fairness_results.keys(), fairness_results.values())
        plt.axhline(0.1, linestyle="--", color="red", label="Fairness threshold")
        plt.ylabel("Difference")
        plt.title("Fairness Disparity Metrics")
        plt.xticks(rotation=45)
        plt.legend()
        save_fig("fairness_disparities")
    else:
        print("Gender disabled — skipping fairness evaluation and plots.")

    # SHAP Analysis
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model.get_booster())
    shap_values = explainer.shap_values(X_test_t)

    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    save_fig("shap_summary")

    # LIME Analysis
    lime_explainer = LimeTabularExplainer(
        training_data=preprocessor.transform(X_train),
        feature_names=feature_names.tolist(),
        class_names=["Bad", "Good"],
        mode="classification",
    )
    exp = lime_explainer.explain_instance(
        preprocessor.transform(X_test)[0],
        model.predict_proba,
        num_features=10,
    )
    exp.as_pyplot_figure()
    plt.title("LIME Local Explanation")
    save_fig("lime_explanation")
    print("Evaluation complete. All applicable visualizations saved.")

if __name__ == "__main__":
    evaluate()
