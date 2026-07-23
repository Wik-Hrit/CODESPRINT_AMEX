import streamlit as st
import requests
import plotly.express as px
import pandas as pd

API_BASE = "http://localhost:5000"
AMEX_BLUE = "#016FD0"

st.set_page_config(page_title="BenefitIQ", page_icon="💳", layout="wide")

st.markdown(f"""
<style>
    .metric-card {{
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid {AMEX_BLUE};
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }}
    h1, h2, h3 {{ color: {AMEX_BLUE}; }}
</style>
""", unsafe_allow_html=True)

st.title("💳 BenefitIQ")
st.caption("ML-powered benefit utilization prediction & personalized nudges")


@st.cache_data(ttl=30)
def fetch_dashboard():
    try:
        r = requests.get(f"{API_BASE}/dashboard", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


metrics, error = fetch_dashboard()

if error:
    st.error(f"Could not reach BenefitIQ API at {API_BASE} ({error}). "
             f"Start it with: `python src/app.py`")
    st.stop()

# --- Metric cards ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Cardholders", f"{metrics['total_users']:,}")
with col2:
    st.metric("Avg. Underutilization", f"{metrics['avg_underutilization_rate']*100:.1f}%")
with col3:
    st.metric("Projected Uplift", f"{metrics['projected_uplift_pct']*100:.0f}%")
with col4:
    recovered_value = metrics['total_users'] * metrics['avg_underutilization_rate'] * 100
    st.metric("Est. Recoverable Value", f"${recovered_value:,.0f}")

st.divider()

# --- Segment pie chart + benefits bar chart ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("User Segments")
    seg_df = pd.DataFrame({
        'Segment': list(metrics['segment_distribution'].keys()),
        'Share': list(metrics['segment_distribution'].values())
    })
    fig_pie = px.pie(seg_df, names='Segment', values='Share',
                      color_discrete_sequence=px.colors.sequential.Blues_r)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("Top Recommended Benefits")
    benefits_df = pd.DataFrame({
        'Benefit': [k.replace('_', ' ').title() for k in metrics['top_recommended_benefits'].keys()],
        'Users': list(metrics['top_recommended_benefits'].values())
    }).sort_values('Users', ascending=True)
    fig_bar = px.bar(benefits_df, x='Users', y='Benefit', orientation='h',
                      color_discrete_sequence=[AMEX_BLUE])
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --- Sample user prediction ---
st.subheader("🔍 Predict for a Cardholder")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        user_id = st.number_input("User ID", value=12345, step=1)
        recency_days = st.slider("Days since last transaction", 0, 180, 15)
    with c2:
        frequency = st.slider("Transaction frequency (90d)", 0, 150, 60)
        monetary = st.number_input("Total spend ($)", value=120000, step=1000)
    with c3:
        age = st.number_input("Age", value=42, step=1)
        income_proxy = st.number_input("Income proxy ($)", value=150000, step=1000)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {
        "user_id": int(user_id),
        "features": {
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary": monetary,
            "age": age,
            "income_proxy": income_proxy,
            "age_days": 300,
            "card_type_gold": 1,
            "card_type_platinum": 0,
        }
    }
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
        r.raise_for_status()
        result = r.json()

        st.success(f"Segment: **{result['segment'].replace('_', ' ').title()}**  |  "
                   f"Utilization probability: **{result['utilization_probability']*100:.1f}%**")

        st.write("**Recommended benefits:**", ", ".join(result['recommended_benefits']))

        st.write("**Personalized nudges:**")
        for nudge in result['nudges']:
            st.info(nudge)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("BenefitIQ · CodeStreet 2026 · Hritwik + Anubhav")
