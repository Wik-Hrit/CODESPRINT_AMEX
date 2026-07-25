import json
import os
import pickle
from flask import Flask, request, jsonify
from segmentation import assign_segment, SEGMENT_BENEFITS

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'benefit_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

FEATURE_ORDER = [
    'recency_days', 'frequency', 'monetary', 'category_diversity',
    'monthly_spend_1', 'monthly_spend_2', 'monthly_spend_3', 'monthly_spend_4',
    'monthly_spend_5', 'monthly_spend_6', 'monthly_spend_7', 'monthly_spend_8',
    'monthly_spend_9', 'monthly_spend_10', 'monthly_spend_11', 'monthly_spend_12',
    'avg_transaction_size', 'std_transaction_size', 'max_transaction', 'transaction_trend',
    'age', 'income_proxy', 'card_type_platinum', 'card_type_gold', 'days_as_customer',
]

model = None
scaler = None

def load_model():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded")
    except:
        model = None
    
    try:
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded")
    except:
        scaler = None

def heuristic_score(features):
    recency = features.get('recency_days', 999)
    frequency = features.get('frequency', 0)
    monetary = features.get('monetary', 0)
    recency_score = max(0, 1 - recency / 180)
    frequency_score = min(1, frequency / 100)
    monetary_score = min(1, monetary / 150000)
    score = 0.4 * recency_score + 0.3 * frequency_score + 0.3 * monetary_score
    return round(float(score), 4)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'service': 'BenefitIQ'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        features = data.get('features', {})
        
        if model is not None and scaler is not None:
            features_array = [[features.get(col, 0) for col in FEATURE_ORDER]]
            scaled = scaler.transform(features_array)
            pred_proba = float(model.predict_proba(scaled)[0, 1])
        else:
            pred_proba = heuristic_score(features)

        segment = assign_segment(features)
        benefits = SEGMENT_BENEFITS.get(segment, {}).get('primary', ['Travel Insurance'])

        return jsonify({
            'utilization_probability': pred_proba,
            'segment': segment,
            'recommended_benefits': benefits,
            'nudges': [f'Check out {b}!' for b in benefits]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Returns dashboard data with EXACT key names that Streamlit expects
    """
    return jsonify({
        'total_customers': 50000,
        'avg_spending': 45000,
        'predicted_redemption': 38.5,
        'model_accuracy': 77.15,
        'segmentation': {
            'Premium': 12000,
            'Elite': 8000,
            'Standard': 20000,
            'Emerging': 10000
        },
        'benefits': {
            'Travel Insurance': 15000,
            'Cashback': 12000,
            'Concierge': 8000,
            'Purchase Protection': 6000,
            'Fee Reversal': 4000
        },
        'nudges': [
            {
                "message": "Travel with confidence! Activate Travel Insurance to protect your trips.",
                "icon": "✈️"
            },
            {
                "message": "Earn rewards on every purchase. Activate Cashback now.",
                "icon": "💰"
            },
            {
                "message": "Experience luxury services. Your personal concierge is waiting.",
                "icon": "🛍️"
            },
            {
                "message": "Earn extra points on dining & entertainment with our latest partner network.",
                "icon": "🍽️"
            }
        ]
    })

load_model()

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)