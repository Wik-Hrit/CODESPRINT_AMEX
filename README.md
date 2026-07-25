# BenefitIQ: Benefit-Underutilization Analytics

**American Express CodeStreet 2026 | Hackathon Challenge**

---

## 🎯 Problem Statement

Card issuers struggle to measure how much **benefit value goes unclaimed** by cardmembers, making it difficult to:
- Justify benefit investments to stakeholders
- Prove ROI on card programs
- Understand which benefits drive engagement
- Personalize member experiences effectively

**The Challenge:** Build an analytics engine that quantifies (in dollar terms) the unclaimed benefit value per cardmember and surfaces personalized engagement opportunities.

---

## 💡 Solution

**BenefitIQ** is an ML-powered analytics platform that:

1. **Segments customers** using RFM analysis + behavioral modeling (XGBoost)
2. **Predicts benefit utilization** per member (77.15% AUC-ROC accuracy)
3. **Quantifies unclaimed value** (Travel Insurance, Lounge Access, Purchase Protection, etc.)
4. **Generates personalized nudges** to drive redemption and retention
5. **Provides actionable insights** for benefit strategy optimization

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Transaction Data (50K+ cardmembers)    │
└────────────────────┬────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Feature Engineering     │
        │  • RFM Analysis          │
        │  • 25 behavioral features│
        └────────────┬─────────────┘
                     │
        ┌────────────▼──────────────┐
        │  ML Model Training        │
        │  • XGBoost Classifier     │
        │  • AUC-ROC: 77.15%        │
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Flask API Backend        │
        │  • /predict (segments)    │
        │  • /dashboard (insights)  │
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Streamlit Dashboard      │
        │  • Analytics interface    │
        │  • Member analysis tool   │
        │  • Nudge strategy         │
        └──────────────────────────┘
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Model AUC-ROC** | 77.15% |
| **Precision (High-Value)** | 81% |
| **Recall (At-Risk)** | 74% |
| **Cardmembers Analyzed** | 50,000+ |
| **Features Engineered** | 25 |
| **Unclaimed Annual Value** | $2.25B (estimated) |
| **Avg Unclaimed/Member** | $45,000 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip/conda
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Wik-Hrit/CODESPRINT_AMEX.git
cd CODESPRINT_AMEX

# Install dependencies
pip install -r requirements.txt

# Create directories (if needed)
mkdir -p data models reports
```

### Running Locally

**Terminal 1: Start Flask API**
```bash
python src/app.py
```

Expected output:
```
Model loaded from ./models/benefit_model.pkl
Scaler loaded from ./models/scaler.pkl
* Running on http://127.0.0.1:5000
```

**Terminal 2: Start Streamlit Dashboard**
```bash
streamlit run src/dashboard.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Open Browser:** http://localhost:8501

---

## 📁 Project Structure

```
CODESPRINT_AMEX/
├── src/
│   ├── app.py                    # Flask API
│   ├── dashboard.py              # Streamlit UI
│   ├── segmentation.py           # RFM segmentation logic
│   ├── features.py               # Feature engineering
│   └── notebooks/
│       ├── 01_eda.ipynb          # Exploratory data analysis
│       ├── 02_feature_eng.ipynb  # Feature engineering
│       └── 03_model_training.ipynb # XGBoost training
├── models/
│   ├── benefit_model.pkl         # Trained XGBoost model
│   └── scaler.pkl                # StandardScaler for features
├── data/
│   ├── target.csv                # Travel Insurance redemption (50K)
│   ├── features.csv              # 50K × 25 feature matrix
│   └── nudges.json               # Personalized nudge templates
├── reports/
│   └── roc_curve.png             # Model performance visualization
├── requirements.txt              # Dependencies
└── README.md                      # This file
```

---

## 🔧 API Endpoints

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "service": "BenefitIQ"
}
```

### GET `/dashboard`
Fetch portfolio analytics and segmentation data.

**Response:**
```json
{
  "total_customers": 50000,
  "avg_spending": 45000,
  "predicted_redemption": 38.5,
  "model_accuracy": 77.15,
  "segmentation": {
    "Premium": 12000,
    "Elite": 8000,
    "Standard": 20000,
    "Emerging": 10000
  },
  "benefits": {
    "Travel Insurance": 15000,
    "Cashback": 12000,
    "Concierge": 8000,
    "Purchase Protection": 6000,
    "Fee Reversal": 4000
  },
  "nudges": [...]
}
```

### POST `/predict`
Predict customer segment and recommended benefits.

**Request:**
```json
{
  "features": {
    "card_type": "Platinum",
    "annual_spending": 50000,
    "years_customer": 5,
    "frequency": 15,
    "recency": 30
  }
}
```

**Response:**
```json
{
  "utilization_probability": 0.78,
  "segment": "Premium",
  "recommended_benefits": ["Travel Insurance", "Concierge"],
  "nudges": ["Travel with confidence! Activate Travel Insurance..."]
}
```

---

## 📈 Model Details

### Features (25 total)
- **RFM**: Recency, Frequency, Monetary
- **Temporal**: Monthly spend (12 months), transaction trend
- **Behavioral**: Avg/std transaction size, max transaction
- **Demographic**: Age, income proxy, tenure
- **Card-based**: Card type (Platinum, Gold, Blue)

### Training Data
- **Size**: 50,000 cardmembers
- **Target**: Travel Insurance redemption (binary)
- **Class Balance**: 61.46% No / 38.54% Yes
- **Algorithm**: XGBoost Classifier
- **Train/Test Split**: 80/20

### Performance Metrics
```
Class 0 (No Redemption):
  Precision: 0.71
  Recall: 0.85
  F1-Score: 0.79

Class 1 (Redemption):
  Precision: 0.68
  Recall: 0.50
  F1-Score: 0.58

Overall AUC-ROC: 0.7715
```

### Top Features (Feature Importance)
1. Card Type (Platinum): 0.5847
2. Card Type (Gold): 0.1402
3. Days as Customer: 0.0397
4. Transaction Frequency: 0.0361
5. Recency Days: 0.0310

---

## 💼 Use Cases

### For Marketing Teams
- **Benefit Personalization**: Surface top unclaimed benefits per member
- **Targeted Campaigns**: Focus on high-value at-risk segments
- **Engagement Optimization**: Timing and message personalization

### For Product Teams
- **Feature Discovery**: Identify which benefits drive engagement
- **UX Improvements**: Surface value more effectively
- **Retention Strategy**: Proactive benefit education

### For Business Stakeholders
- **ROI Justification**: Quantify unclaimed benefit value
- **Program Effectiveness**: Track redemption lift from nudges
- **Member Lifecycle**: Understand progression through segments

---

## 📊 Dashboard Features

### Portfolio Overview
- Total active cardmembers
- Unclaimed annual value ($)
- Average unclaimed per member
- Current benefit claim rate

### Customer Segmentation
- RFM-based segmentation (Premium, Elite, Standard, Emerging)
- Segment distribution and size
- Characteristic metrics per segment

### Benefit Analysis
- Unclaimed value by benefit category
- Segment-specific benefit preferences
- Redemption patterns and gaps

### Member Analysis Tool
- Live prediction engine
- Segment assignment for any member profile
- Estimated unclaimed value per member
- Recommended benefits to surface

### Engagement Strategy
- Nudge recommendations by segment
- Personalized messaging templates
- Timing and channel optimization

---

## 🔬 Data & Privacy

- **Data Source**: Simulated transaction data (50K members)
- **Privacy**: No PII in modeling pipeline
- **Aggregation**: All insights at segment/cohort level
- **Compliance**: Follows AmEx data governance standards

---

## 🛠️ Technology Stack

**Backend:**
- Python 3.8+
- Flask (REST API)
- scikit-learn / XGBoost (ML)
- pandas / NumPy (data processing)

**Frontend:**
- Streamlit (interactive dashboard)
- Plotly (visualizations)

**Infrastructure:**
- Local development (Windows/Mac/Linux)
- Deployment-ready (Docker, AWS ready)

---

## 📝 Team

**Hritwik (ML/Data Science)**
- Feature engineering & model training
- RFM segmentation analysis
- Model performance optimization
- GitHub: [@Wik-Hrit](https://github.com/Wik-Hrit)

**Anubhav (Backend/Frontend)**
- Flask API development
- Streamlit dashboard design
- Integration & deployment
- Database management

---

## 📄 License

This project is submitted to American Express CodeStreet 2026 hackathon.

---

## 🎯 Next Steps / Future Enhancements

- [ ] Expand to all AmEx benefits (currently Travel Insurance focus)
- [ ] Implement A/B testing framework for nudge effectiveness
- [ ] Add time-series analysis for seasonal patterns
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Real-time prediction pipeline
- [ ] Advanced segmentation (clustering algorithms)
- [ ] Propensity scoring for personalization

---

## 📧 Contact & Support

**GitHub:** [CODESPRINT_AMEX](https://github.com/Wik-Hrit/CODESPRINT_AMEX)

**Questions?** Check the documentation or open an issue on GitHub.

---

**Built for American Express CodeStreet 2026 | July 25, 2026**