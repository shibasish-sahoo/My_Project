from pathlib import Path
import json
import pickle
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROCESSED_DIR / "final_combined_news.csv"


def build_features(df):
    model_df = df.copy()
    model_df["text"] = (
        model_df["title"].fillna("").astype(str) + " " + model_df["content"].fillna("").astype(str)
    )
    return model_df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), "text"),
            (
                "structured",
                OneHotEncoder(handle_unknown="ignore"),
                [
                    "source",
                    "dataset_type",
                    "geo_risk_level",
                    "sentiment",
                    "energy_risk",
                    "is_geopolitical",
                    "is_financial",
                ],
            ),
        ]
    )


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "complement_nb": ComplementNB(),
    }


def evaluate_model(name, estimator, x_train, x_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
    )

    train_start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = pipeline.predict(x_test)
    predict_time = time.perf_counter() - predict_start

    return {
        "name": name,
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "training_time_seconds": train_time,
        "prediction_time_seconds": predict_time,
        "predictions": predictions,
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training dataset: {DATA_PATH}")

    df = build_features(pd.read_csv(DATA_PATH))
    if df["investment_recommendation"].nunique() < 2:
        raise RuntimeError("Training requires at least two recommendation classes.")

    stratify_target = df["investment_recommendation"] if df["investment_recommendation"].value_counts().min() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        df[
            [
                "text",
                "source",
                "dataset_type",
                "geo_risk_level",
                "sentiment",
                "energy_risk",
                "is_geopolitical",
                "is_financial",
            ]
        ],
        df["investment_recommendation"],
        test_size=0.25,
        random_state=42,
        stratify=stratify_target,
    )

    results = []
    for name, estimator in candidate_models().items():
        results.append(evaluate_model(name, estimator, x_train, x_test, y_train, y_test))

    best = max(results, key=lambda item: (item["f1_weighted"], item["accuracy"]))

    with open(ARTIFACTS_DIR / "best_model.pkl", "wb") as model_file:
        pickle.dump(best["pipeline"], model_file)

    comparison = [
        {
            "name": result["name"],
            "accuracy": round(result["accuracy"], 4),
            "f1_weighted": round(result["f1_weighted"], 4),
            "training_time_seconds": round(result["training_time_seconds"], 4),
            "prediction_time_seconds": round(result["prediction_time_seconds"], 4),
        }
        for result in results
    ]

    evaluation_payload = {
        "dataset_path": str(DATA_PATH),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "best_model": best["name"],
        "accuracy": round(best["accuracy"], 4),
        "f1_weighted": round(best["f1_weighted"], 4),
        "training_time_seconds": round(best["training_time_seconds"], 4),
        "prediction_time_seconds": round(best["prediction_time_seconds"], 4),
        "model_comparison": comparison,
        "classification_report": best["report"],
        "improvement_notes": [
            "Increase verified-source coverage and historical depth.",
            "Replace rule-generated labels with expert-annotated targets.",
            "Evaluate transformer-based text encoders for richer geopolitical context.",
        ],
    }

    with open(ARTIFACTS_DIR / "model_evaluation.json", "w", encoding="utf-8") as output_file:
        json.dump(evaluation_payload, output_file, indent=2)

    prediction_samples = x_test.copy()
    prediction_samples["actual"] = y_test.values
    prediction_samples["predicted"] = best["predictions"]
    prediction_samples.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    print("Model training and evaluation completed")
    print(json.dumps(evaluation_payload, indent=2))


if __name__ == "__main__":
    main()
