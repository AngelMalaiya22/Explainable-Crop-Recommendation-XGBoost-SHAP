from flask import Flask, render_template, request
import joblib
import json
import os

import numpy as np
import pandas as pd
import shap

app = Flask(__name__)

# ---------------------------------------------------------
# Load model artifacts once, at startup - not per-request
# ---------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

model = joblib.load(os.path.join(MODEL_DIR, 'crop_recommendation_xgb.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'crop_label_encoder.pkl'))

with open(os.path.join(MODEL_DIR, 'feature_order.json')) as f:
    FEATURE_ORDER = json.load(f)   # e.g. ['N', 'P', 'K', 'temp', 'ph', 'humidity']

# SHAP explainer - built once at startup (same TreeExplainer used in the notebook)
explainer = shap.TreeExplainer(model)

# Human-readable labels + simple input constraints for the form
FIELD_INFO = {
    'N':        {'label': 'Nitrogen (N)',        'min': 0,   'max': 200, 'step': 0.1},
    'P':        {'label': 'Phosphorus (P)',       'min': 0,   'max': 150, 'step': 0.1},
    'K':        {'label': 'Potassium (K)',        'min': 0,   'max': 210, 'step': 0.1},
    'temp':     {'label': 'Temperature (°C)',     'min': 0,   'max': 50,  'step': 0.1},
    'ph':       {'label': 'Soil pH',               'min': 0,   'max': 14,  'step': 0.01},
    'humidity': {'label': 'Humidity (%)',          'min': 0,   'max': 100, 'step': 0.1},
}


def explain_prediction(input_df, class_idx):
    """
    Local SHAP explanation for ONE prediction, for the predicted class only.
    Returns a list of dicts (one per feature) sorted by absolute impact.
    Positive shap = pushes the model TOWARD the predicted crop.
    Negative shap = pushes the model AWAY from the predicted crop.
    """
    sv = explainer.shap_values(input_df)

    # Handle both SHAP output shapes (list of arrays vs one 3D array)
    if isinstance(sv, list):
        contrib = np.asarray(sv[class_idx])[0]        # (n_features,)
    else:
        sv = np.asarray(sv)
        contrib = sv[0, :, class_idx] if sv.ndim == 3 else sv[0]

    max_abs = float(np.max(np.abs(contrib))) or 1.0
    rows = []
    for feat, val, c in zip(FEATURE_ORDER, input_df.iloc[0].tolist(), contrib):
        rows.append({
            'label': FIELD_INFO.get(feat, {}).get('label', feat),
            'value': round(float(val), 2),
            'shap': round(float(c), 4),
            'width': round(abs(float(c)) / max_abs * 100, 1),
            'positive': bool(c >= 0),
        })
    rows.sort(key=lambda r: abs(r['shap']), reverse=True)
    return rows


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    explanation = None
    explain_error = None
    error = None
    form_values = {}

    if request.method == 'POST':
        try:
            # Collect + validate every expected field
            for col in FEATURE_ORDER:
                raw = request.form.get(col, '').strip()
                if raw == '':
                    raise ValueError(f"Please fill in {FIELD_INFO.get(col, {}).get('label', col)}.")
                form_values[col] = float(raw)

            # Build the feature row in the EXACT order the model was trained on
            input_df = pd.DataFrame([[form_values[col] for col in FEATURE_ORDER]],
                                    columns=FEATURE_ORDER)

            pred_encoded = model.predict(input_df)
            class_idx = int(pred_encoded[0])
            prediction = label_encoder.inverse_transform(pred_encoded)[0]

            # Explanation is optional: if SHAP fails, the prediction still shows
            try:
                explanation = explain_prediction(input_df, class_idx)
            except Exception as ex:
                explain_error = f"Explanation unavailable: {ex}"

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Something went wrong while predicting: {e}"

    return render_template(
        'index.html',
        feature_order=FEATURE_ORDER,
        field_info=FIELD_INFO,
        prediction=prediction,
        explanation=explanation,
        explain_error=explain_error,
        error=error,
        form_values=form_values,
    )


if __name__ == '__main__':
    app.run(debug=True)
