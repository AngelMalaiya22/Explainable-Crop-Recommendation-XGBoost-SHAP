# Crop Recommendation — Flask App

## Setup

1. **Copy your saved model files** into `saved_models/` in this folder:
   - `crop_recommendation_xgb.pkl`
   - `crop_label_encoder.pkl`
   - `feature_order.json`

   (These are the files you already saved in your notebook's `09_Save Model` section.)

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```
   python app.py
   ```

4. Open **http://127.0.0.1:5000/** in your browser.

## Folder structure

```
crop_recommendation_app/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── saved_models/
    ├── crop_recommendation_xgb.pkl      <- copy from your notebook output
    ├── crop_label_encoder.pkl           <- copy from your notebook output
    └── feature_order.json               <- copy from your notebook output
```

## How it works

- `app.py` loads the model, label encoder, and feature order **once** at startup (not per request — faster).
- The form fields are generated dynamically from `feature_order.json`, so the input order always matches what the model was trained on — no risk of misaligned columns.
- Predictions are decoded back to crop names using the saved `label_encoder`.
- Basic validation: empty fields and non-numeric input are caught and shown as an error message instead of crashing the app.

## Deploying online (same as your Module 1 approach)

Once this runs locally, you can deploy it the same way you deployed Module 1 (e.g. Render):
1. Push this folder (including the `saved_models/` files) to a GitHub repo.
2. Add a `Procfile` with: `web: gunicorn app:app`
3. Add `gunicorn` to `requirements.txt`.
4. Connect the repo to Render (or your host of choice) as a Web Service.
