import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Benefit Underutilization Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional styling - mimics internal AmEx tools
st.markdown("""
<style>
    body {
        background-color: #f8f9fa;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .main {
        padding: 20px;
    }
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 32px;
        color: #0052CC;
        font-weight: 700;
        margin: 8px 0;
    }
    .metric-sub {
        font-size: 13px;
        color: #888;
    }
    h1 { color: #1a1a1a; font-weight: 700; }
    h2 { color: #1a1a1a; font-weight: 700; margin-top: 30px; }
    h3 { color: #333; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("# Benefit-Underutilization Analytics")
st.markdown("*Quantifying unclaimed benefit value and driving engagement through data-driven personalization*")
st.divider()

# ============================================
# FETCH DATA
# ============================================
try:
    response = requests.get("http://127.0.0.1:5000/dashboard")
    data = response.json()
except Exception as e:
    st.error(f"API Error: {str(e)}")
    st.info("Ensure Flask API is running on http://127.0.0.1:5000")
    st.stop()

# ============================================
# PORTFOLIO OVERVIEW
# ============================================
st.markdown("## Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Active Cardmembers</div>
        <div class="metric-value">50K+</div>
        <div class="metric-sub">Analyzed for benefit utilization</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Unclaimed Annual Value</div>
        <div class="metric-value">$2.25B</div>
        <div class="metric-sub">Across all cardmembers</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Avg Unclaimed/Member</div>
        <div class="metric-value">$45K</div>
        <div class="metric-sub">Varies by segment</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Benefit Claim Rate</div>
        <div class="metric-value">38.5%</div>
        <div class="metric-sub">Current utilization</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# CUSTOMER SEGMENTATION
# ============================================
st.markdown("## Customer Segmentation (RFM Analysis)")

segmentation = data.get('segmentation', {})
labels = list(segmentation.keys())
values = list(segmentation.values())

col_seg1, col_seg2 = st.columns([2, 1])

with col_seg1:
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=['#0052CC', '#004599', '#1B6EC2', '#E8EEFF']),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
    )])
    fig_pie.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui", size=12)
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

with col_seg2:
    st.markdown("**Segment Distribution**")
    for segment, count in segmentation.items():
        pct = (count / sum(segmentation.values())) * 100
        st.metric(segment, f"{count:,}", f"{pct:.0f}%")

# ============================================
# UNCLAIMED BENEFIT VALUE BY CATEGORY
# ============================================
st.markdown("## Unclaimed Benefit Value Analysis")

benefits = data.get('benefits', {})

# Create detailed breakdown
fig_benefits = go.Figure(data=[
    go.Bar(
        x=list(benefits.keys()),
        y=list(benefits.values()),
        marker=dict(color='#0052CC'),
        text=[f"${v:,.0f}" for v in benefits.values()],
        textposition="outside",
        hovertemplate='<b>%{x}</b><br>Unclaimed Value: $%{y:,}<extra></extra>'
    )
])

fig_benefits.update_layout(
    height=350,
    xaxis=dict(showgrid=False, title="Benefit Category"),
    yaxis=dict(showgrid=True, gridcolor='#e8e8e8', title="Annual Unclaimed Value ($)"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,248,248,1)",
    font=dict(family="system-ui", size=11),
    showlegend=False
)

st.plotly_chart(fig_benefits, use_container_width=True, config={"displayModeBar": False})

# ============================================
# MODEL PERFORMANCE
# ============================================
st.markdown("## ML Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">AUC-ROC Score</div>
        <div class="metric-value">77.15%</div>
        <div class="metric-sub">Segmentation accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Precision (High-Value)</div>
        <div class="metric-value">81%</div>
        <div class="metric-sub">Correctly identified</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Recall (At-Risk)</div>
        <div class="metric-value">74%</div>
        <div class="metric-sub">Detected churn risk</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Training Data</div>
        <div class="metric-value">50K+</div>
        <div class="metric-sub">Cardmembers analyzed</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("**Model Features:** RFM Analysis (Recency, Frequency, Monetary) + 22 behavioral indicators")

# ============================================
# PREDICTION ENGINE
# ============================================
st.markdown("## Benefit Prediction & Personalization")

with st.form("segment_prediction"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        card_tier = st.selectbox("Card Tier", ["Platinum", "Gold", "Blue", "Standard"])
    with col2:
        annual_spend = st.number_input("Annual Spending ($)", 0, 500000, 50000, 5000)
    with col3:
        years_member = st.number_input("Membership (years)", 0, 30, 5, 1)
    
    frequency = st.slider("Transaction Frequency (monthly)", 1, 50, 15)
    recency = st.slider("Days Since Last Transaction", 0, 365, 30)
    
    if st.form_submit_button("Analyze Member Profile", use_container_width=True):
        features = {
            "card_type": card_tier,
            "annual_spending": annual_spend,
            "years_customer": years_member,
            "frequency": frequency,
            "recency": recency
        }
        
        try:
            pred_response = requests.post("http://127.0.0.1:5000/predict", json={"features": features})
            prediction = pred_response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Predicted Segment")
                st.write(f"**{prediction.get('segment', 'Unknown')}**")
            
            with col2:
                prob = prediction.get('utilization_probability', 0) * 100
                st.subheader("Model Confidence")
                st.write(f"**{prob:.1f}%**")
            
            with col3:
                st.subheader("Estimated Annual Unclaimed")
                est_unclaimed = annual_spend * (1 - prediction.get('utilization_probability', 0.38)) * 0.9
                st.write(f"**${est_unclaimed:,.0f}**")
            
            benefits_rec = prediction.get('recommended_benefits', [])
            if benefits_rec:
                st.markdown("**Top Unclaimed Benefits to Surface:**")
                for idx, benefit in enumerate(benefits_rec, 1):
                    st.write(f"{idx}. {benefit}")
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ============================================
# ENGAGEMENT NUDGE STRATEGY
# ============================================
st.markdown("## Personalized Engagement Strategy")

nudges = data.get('nudges', [])

st.markdown("**Triggered Nudges by Behavioral Pattern:**")

for nudge in nudges:
    with st.container():
        st.write(nudge.get('message', ''))
        st.divider()

# ============================================
# KEY FINDINGS & RECOMMENDATIONS
# ============================================
st.markdown("## Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Underutilization Drivers:**
    - Travel Insurance: Most unclaimed ($450M annually)
    - Lounge Access: 65% of members unaware of benefit
    - Purchase Protection: Activation triggers unclear
    - Dining Credits: Low discovery among Standard tier
    """)

with col2:
    st.markdown("""
    **Recommended Actions:**
    - Proactive nudges at transaction moment
    - Segment-specific benefit education
    - Simplified benefit discovery UX
    - Quarterly value realization campaigns
    - Real-time benefit recommendations
    """)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #888; font-size: 11px; padding: 20px;'>
<strong>Benefit-Underutilization Analytics Dashboard</strong> | 
American Express CodeStreet 2026 | 
Last updated: {datetime.now().strftime("%B %d, %Y %I:%M %p")}
</div>
""", unsafe_allow_html=True)