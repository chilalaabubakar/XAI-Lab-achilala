import os
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import config

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    df = config.load_data()

    # Gender distribution logging
    if config.WITH_GENDER:
        gender_counts = df["Sex"].value_counts()
        print("+-+" * 15)
        print("Gender distribution in dataset:")
        print("+-+" * 15)
        for gender, count in gender_counts.items():
            print(f"{gender}: {count}")

    # Prepare features and target
    drop_cols = [config.TARGET]
    if not config.WITH_GENDER:
        drop_cols.append("Sex")

    X = df.drop(columns=drop_cols)
    y = config.encode_target(df[config.TARGET])

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLS,
            ),
            ("num", "passthrough", config.NUMERICAL_COLS),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    # Model definition
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # MLflow Tracking
    mlflow.set_experiment("Credit_Risk_Assessment")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, config.MODEL_PATH)
        
        # Log model and parameters
        mlflow.sklearn.log_model(pipeline, "xgboost_pipeline")
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "with_gender": config.WITH_GENDER
        })
        print(f"Model successfully saved to {config.MODEL_PATH}")

    return pipeline, X_test, y_test

if __name__ == "__main__":
    train()
