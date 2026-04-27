from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ENERGY_KEYWORDS = [
    "oil",
    "gas",
    "energy",
    "pipeline",
    "crude",
    "oil tanker",
    "shipping",
    "energy crisis",
]


def has_energy_risk(text):
    text = str(text).lower()
    return any(word in text for word in ENERGY_KEYWORDS)


def investment_recommendation(row):
    risk = row["geo_risk_level"]
    sentiment = row["sentiment"]
    energy = row["energy_risk"]

    if risk == "High":
        if energy:
            return "Strong Buy: Gold, Oil & Energy Stocks | Avoid Equities"
        return "Buy Gold & Defence Stocks | Avoid Equities"

    if risk == "Medium":
        if sentiment == "Positive":
            return "Selective Equity Buy | Hold Gold"
        return "Hold Gold | Avoid High-Risk Equities"

    if sentiment == "Positive":
        return "Buy Equities, Banking, IT"
    if sentiment == "Negative":
        return "Hold Cash | Defensive Stocks"
    return "Market Neutral | Hold Positions"


def process_file(path):
    df = pd.read_csv(path)
    df["energy_risk"] = df.apply(
        lambda row: has_energy_risk(row.get("title", "")) or has_energy_risk(row.get("content", "")),
        axis=1,
    )
    df["investment_recommendation"] = df.apply(investment_recommendation, axis=1)
    output_name = path.stem.replace("_with_risk", "").replace("combined_news", "final_combined_news")
    output_name = output_name.replace("geopolitical_news", "final_geopolitical_news")
    output_name = output_name.replace("financial_news", "final_financial_news")
    output_path = PROCESSED_DIR / f"{output_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"Investment recommendation engine completed for {path.name} -> {output_path.name}")


def main():
    for name in ("geopolitical_news_with_risk.csv", "financial_news_with_risk.csv", "combined_news_with_risk.csv"):
        path = PROCESSED_DIR / name
        if path.exists():
            process_file(path)

    final_combined_path = PROCESSED_DIR / "final_combined_news.csv"
    if final_combined_path.exists():
        pd.read_csv(final_combined_path).to_csv(BASE_DIR / "final_investment_recommendations.csv", index=False)


if __name__ == "__main__":
    main()
