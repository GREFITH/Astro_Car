# gradio_app.py
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import gradio as gr

HERE = Path(__file__).parent

# --- File names you must have in repo ---
MODEL_FILE = HERE / "model.pkl"           # trained pipeline (preprocessor + model)
META_FILE = HERE / "model.meta.json"      # optional: contains features order, numeric/categorical lists
CAR_IMAGE = "../static/ "       # image used in UI (relative path)

# --- Load model ---
if not MODEL_FILE.exists():
    raise SystemExit("Missing model.pkl — run training script and put model.pkl into repo root")

model = joblib.load(MODEL_FILE)

# --- Load metadata (optional but recommended) ---
if META_FILE.exists():
    meta = json.loads(META_FILE.read_text())
    FEATURES = meta.get("features", None)
    NUMERIC = meta.get("numeric_features", [])
    CATEGORICAL = meta.get("categorical_features", [])
else:
    # Try to infer feature names from model if it's a pipeline with named steps
    FEATURES = None
    NUMERIC = []
    CATEGORICAL = []

# Fallback: define a sensible default feature order used by the frontend.
# Adjust these names to match whatever features the model expects if your meta is missing.
if not FEATURES:
    FEATURES = ["brand", "model", "year", "mileage", "fuel_type", "engine_cc", "power_hp"]
    NUMERIC = ["year", "mileage", "engine_cc", "power_hp"]
    CATEGORICAL = ["brand", "model", "fuel_type"]

# --- Helper: coerce and build DataFrame matching FEATURES order ---
def build_input_df(payload_dict):
    # payload_dict: mapping component_name->value
    row = {}
    for f in FEATURES:
        v = payload_dict.get(f, None)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            # default for numeric -> 0 or median-like safe value; for categorical -> 'unknown'
            if f in NUMERIC:
                row[f] = 0.0
            else:
                row[f] = "unknown"
            continue
        # coerce numeric
        if f in NUMERIC:
            try:
                row[f] = float(v)
            except Exception:
                row[f] = 0.0
        else:
            row[f] = v
    X = pd.DataFrame([row], columns=FEATURES)
    return X

# --- Prediction function used by Gradio ---
def predict(brand, model_name, year, mileage, fuel_type, engine_cc, power_hp):
    """
    The function signature must match the components order below.
    Returns: price_str, debug (optional)
    """
    payload = {
        "brand": brand,
        "model": model_name,
        "year": year,
        "mileage": mileage,
        "fuel_type": fuel_type,
        "engine_cc": engine_cc,
        "power_hp": power_hp,
    }
    X = build_input_df(payload)
    try:
        raw_pred = model.predict(X)[0]
        price = float(raw_pred) if raw_pred is not None else 0.0
        if price < 0:
            price = 0.0
        price_str = f"₹ {int(round(price)):,}"
        # Optional: small explanation using simple model attributes if available
        debug = f"Model output: {price:.2f}"
        return price_str, debug
    except Exception as e:
        return "Prediction error", f"Error: {str(e)}"

# --- Gradio UI layout ---
def build_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=6):
                # Visual / image + output text
                car_img = gr.Image(value=CAR_IMAGE, interactive=False, label="Astro Car")
                predicted_text = gr.Markdown("### Predicted price\n`—`", elem_id="predicted_price_md")
                debug_text = gr.Textbox(value="", label="Debug / Info", interactive=False)

            with gr.Column(scale=4):
                gr.Markdown("## Input features\nFeed Astro Cars with car specs")
                brand = gr.Dropdown(
                    choices=[
                        "Other", "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra",
                        "Kia", "Ford", "Volkswagen", "BMW", "Mercedes"
                    ],
                    value="Other",
                    label="Brand"
                )
                model_name = gr.Textbox(placeholder="e.g. i20 / City / Creta", label="Model")
                year = gr.Number(value=2018, label="Model year")
                mileage = gr.Number(value=35000, label="Mileage (km)")
                fuel_type = gr.Dropdown(choices=["Other", "Petrol", "Diesel", "CNG", "Electric"], value="Other", label="Fuel type")
                engine_cc = gr.Number(value=1200, label="Engine capacity (cc)")
                power_hp = gr.Slider(minimum=40, maximum=500, value=120, step=1, label="Engine power (HP)")

                with gr.Row():
                    predict_btn = gr.Button("Predict Price")
                    reset_btn = gr.Button("Reset")

        # Hook up button
        predict_btn.click(
            fn=predict,
            inputs=[brand, model_name, year, mileage, fuel_type, engine_cc, power_hp],
            outputs=[predicted_text, debug_text],
            queue=False
        )

        # Reset action
        def _reset():
            return gr.Dropdown.update(value="Other"), "", 2018, 35000, "Other", 1200, 120

        reset_btn.click(
            fn=_reset,
            inputs=[],
            outputs=[brand, model_name, year, mileage, fuel_type, engine_cc, power_hp]
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
