import json
import os
import pickle

from flask import Flask, request, jsonify

from segmentation import assign_segment, SEGMENT_BENEFITS

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'benefit_model.pkl')
NUDGES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'nudges.json')

# The 25 features in the exact order the model expects.
# Keep this in sync with src/features.py's output columns (minus user_id).
FEATURE_ORDER = [
    'recency_days', 'frequency', 'monetary', 'category_diversity',
    'monthly_spend_1', 'monthly_spend_2', 'monthly_spend_3', 'monthly_spend_4',
    'monthly_spend_5', 'monthly_spend_6', 'monthly_spend_7', 'monthly_spend_8',
    'monthly_spend_9', 'monthly_spend_10', 'monthly_spend_11', 'monthly_spend_12',
    'avg_transaction', 'std_transaction', 'max_transaction', 'transaction_trend',
    'age', 'income_proxy', 'card_type_gold', 'card_type_platinum', 'age_days',
]

model = None
nudge_templates = {}


def load_model():
    """Load the trained model if it exists. API still works without it
    (falls back to a heuristic score) so Anubhav isn't blocked waiting on Hritwik."""
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from", MODEL_PATH)
    else:
        model = None
        print(f"No model found at {MODEL_PATH} yet - /predict will use a heuristic score.")


def load_nudges():
    global nudge_templates
    with open(NUDGES_PATH) as f:
        nudge_templates = json.load(f)


def heuristic_score(features):
    """Fallback utilization-probability estimate, used until the real model is dropped in.
    Roughly: recent + frequent + high spend -> higher probability of using benefits."""
    recency = features.get('recency_days', 999)
    frequency = features.get('frequency', 0)
    monetary = features.get('monetary', 0)

    recency_score = max(0, 1 - recency / 180)
    frequency_score = min(1, frequency / 100)
    monetary_score = min(1, monetary / 150000)

    score = 0.4 * recency_score + 0.3 * frequency_score + 0.3 * monetary_score
    return round(float(score), 4)


def select_benefits_for_segment(segment):
    return SEGMENT_BENEFITS.get(segment, SEGMENT_BENEFITS['at_risk'])['primary']


def generate_nudges(benefits):
    """Pull real templates from data/nudges.json, keyed by benefit name."""
    nudges = []
    for benefit in benefits[:2]:
        key = benefit.lower().replace(' ', '_')
        options = nudge_templates.get(key)
        if options:
            nudges.append(options[0].replace('{trips}', '3'))
        else:
            nudges.append(f"Activate {benefit} today.")
    return nudges


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'service': 'BenefitIQ',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input: {
        "user_id": 12345,
        "features": { "recency_days": 15, "frequency": 50, "monetary": 100000, ... }
    }

    Output: {
        "user_id": 12345,
        "utilization_probability": 0.78,
        "segment": "high_value",
        "recommended_benefits": ["Travel Insurance", "Purchase Protection"],
        "nudges": ["You've traveled 3x this year..."]
    }
    """
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        features = data.get('features', {})

        if model is not None:
            features_array = [features.get(col, 0) for col in FEATURE_ORDER]
            pred_proba = float(model.predict_proba([features_array])[0, 1])
        else:
            pred_proba = heuristic_score(features)

        segment = assign_segment(features)
        benefits = select_benefits_for_segment(segment)
        nudges = generate_nudges(benefits)

        return jsonify({
            'user_id': user_id,
            'utilization_probability': pred_proba,
            'segment': segment,
            'recommended_benefits': benefits,
            'nudges': nudges
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Aggregated dashboard metrics, computed from data/features.csv if present,
    otherwise falls back to representative placeholder numbers."""
    features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')

    if os.path.exists(features_path):
        import pandas as pd
        df = pd.read_csv(features_path)
        segments = df.apply(lambda row: assign_segment(row.to_dict()), axis=1)
        seg_counts = segments.value_counts(normalize=True).round(3).to_dict()
        total_users = len(df)
        avg_underutilization = round(1 - min(1, df['frequency'].mean() / 100), 3)
    else:
        total_users = 50000
        avg_underutilization = 0.65
        seg_counts = {'high_value': 0.30, 'at_risk': 0.35, 'new': 0.20, 'dormant': 0.15}

    return jsonify({
        'total_users': total_users,
        'avg_underutilization_rate': avg_underutilization,
        'segment_distribution': seg_counts,
        'projected_uplift_pct': 0.45,
        'top_recommended_benefits': {
            'travel_insurance': 15000,
            'purchase_protection': 12000,
            'concierge': 8000,
            'fee_reversal': 6000
        }
    })


load_model()
load_nudges()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
