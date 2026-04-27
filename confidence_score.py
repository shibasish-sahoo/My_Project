from pathlib import Path
import pickle

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATA_PATH = PROCESSED_DIR / "final_combined_news.csv"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["text"] = df["title"].fillna("").astype(str) + " " + df["content"].fillna("").astype(str)

    feature_frame = df[
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
    ]

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The saved model does not support predict_proba.")

    probabilities = model.predict_proba(feature_frame).max(axis=1)
    df["confidence_score_%"] = (probabilities * 100).round(2)

    output_path = ARTIFACTS_DIR / "final_with_confidence.csv"
    df.to_csv(output_path, index=False)

    print(f"Confidence scores added -> {output_path}")
    print(df[["investment_recommendation", "confidence_score_%"]].head())


if __name__ == "__main__":
    main()
