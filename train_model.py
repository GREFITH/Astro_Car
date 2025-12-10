#!/usr/bin/env python3
"""
Train a simple model and save pipeline + meta.
Usage:
    python train_model.py --csv car_dataset.csv --model model.pkl
If no --csv provided, tries ./car_dataset.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Helper to construct OneHotEncoder handling different sklearn versions:
def make_onehot(**kwargs):
    # scikit-learn 1.2+ uses sparse_output; older uses sparse
    # prefer sparse_output, fallback to sparse
    try:
        return OneHotEncoder(**{**kwargs, **{"sparse_output": False}})
    except TypeError:
        return OneHotEncoder(**{**kwargs, **{"sparse": False}})

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    # candidate names
    target_candidates = ["price", "price(inr)", "selling_price", "selling price", "sellingprice", "price_inr"]
    num_candidates = {
        "year": ["year", "model_year", "yr"],
        "mileage": ["mileage", "km_driven", "km", "kilometers", "kilometres"],
        "engine_cc": ["engine_cc", "engine capacity", "engine_cc", "displacement", "enginecc"],
        "power_hp": ["power_hp", "power", "hp"],
    }
    cat_candidates = {
        "brand": ["brand", "make", "manufacturer"],
        "model": ["model", "name"],
        "fuel_type": ["fuel", "fuel_type"]
    }

    # find target
    target = None
    for t in target_candidates:
        for c in df.columns:
            if c.lower() == t:
                target = c
                break
        if target:
            break

    # fallback: look for any numeric-looking column named *price* substring
    if target is None:
        for c in df.columns:
            if "price" in c.lower():
                target = c
                break

    numeric_found = []
    for canonical, cand_list in num_candidates.items():
        for c in df.columns:
            if c.lower() in cand_list:
                numeric_found.append(c)
                break

    categorical_found = []
    for canonical, cand_list in cat_candidates.items():
        for c in df.columns:
            if c.lower() in cand_list:
                categorical_found.append(c)
                break

    # If model name present, include it too
    # Return best guess list
    return target, numeric_found, categorical_found

def main(csv_path: Path, model_out: Path):
    print("Loading", csv_path)
    df = pd.read_csv(csv_path)

    print("Columns:", list(df.columns))

    target_col, numeric_cols, categorical_cols = detect_columns(df)
    if target_col is None:
        raise SystemExit("Could not detect target column (price). Please check CSV headers.")

    print("Detected target column:", target_col)
    # For robustness, select columns that exist in df
    # Ensure numeric columns are actually numeric
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # If dataset uses different column names (like 'selling_price'), adapt to canonical names later
    # For our pipeline we will rename columns to canonical short names
    rename_map = {}
    canonical_numeric = []
    for c in numeric_cols:
        if "year" in c.lower():
            rename_map[c] = "year"
            canonical_numeric.append("year")
        elif any(x in c.lower() for x in ["mileage", "km"]):
            rename_map[c] = "mileage"
            canonical_numeric.append("mileage")
        elif "engine" in c.lower() or "cc" in c.lower() or "displacement" in c.lower():
            rename_map[c] = "engine_cc"
            canonical_numeric.append("engine_cc")
        elif any(x in c.lower() for x in ["power", "hp"]):
            rename_map[c] = "power_hp"
            canonical_numeric.append("power_hp")
        else:
            # keep as is
            canonical_numeric.append(c)

    canonical_cats = []
    for c in categorical_cols:
        if "brand" in c.lower() or "make" in c.lower() or "manufacturer" in c.lower():
            rename_map[c] = "brand"
            canonical_cats.append("brand")
        elif "model" in c.lower() or "name" in c.lower():
            rename_map[c] = "model"
            canonical_cats.append("model")
        elif "fuel" in c.lower():
            rename_map[c] = "fuel_type"
            canonical_cats.append("fuel_type")
        else:
            canonical_cats.append(c)

    # Always ensure we have at least brand + year + mileage + engine_cc + power_hp if possible
    # If missing some numeric columns, we will still train using available features
    df = df.rename(columns=rename_map)

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Prepare X and y
    y = df[target_col].astype(float)

    # Build feature list: combine numeric & categorical if present
    numeric_use = [c for c in ["year", "mileage", "engine_cc", "power_hp"] if c in df.columns]
    categorical_use = [c for c in ["brand", "model", "fuel_type"] if c in df.columns]

    X = df[numeric_use + categorical_use].copy()

    # Fill missing numeric with median, categorical with 'unknown'
    for c in numeric_use:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in categorical_use:
        X[c] = X[c].fillna("unknown").astype(str)

    print("Numeric features:", numeric_use)
    print("Categorical features:", categorical_use)

    # Build preprocessing
    transformers = []
    if numeric_use:
        transformers.append(("num", StandardScaler(), numeric_use))
    if categorical_use:
        transformers.append(("cat", make_onehot(handle_unknown="ignore"), categorical_use))

    if transformers:
        pre = ColumnTransformer(transformers=transformers, remainder="drop")
    else:
        pre = "passthrough"

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("Training model...")
    pipe.fit(X_train, y_train)
    print("Training Completed!")
    print("Train R2:", pipe.score(X_train, y_train))
    print("Test R2:", pipe.score(X_test, y_test))

    # Save model and metadata (features order)
    dump(pipe, model_out)
    meta = {
        "target": target_col,
        "numeric_features": numeric_use,
        "categorical_features": categorical_use,
        "features": numeric_use + categorical_use
    }
    meta_path = model_out.with_suffix(".meta.json")
    import json
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Model saved as {model_out}")
    print(f"Metadata saved as {meta_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="car_dataset.csv", help="CSV dataset path")
    p.add_argument("--model", type=str, default="model.pkl", help="Output model path")
    args = p.parse_args()
    main(Path(args.csv), Path(args.model))
