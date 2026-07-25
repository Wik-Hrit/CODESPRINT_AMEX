"""
Segmentation logic for BenefitIQ.
Assigns customers to segments based on RFM features and other characteristics.
Works without requiring data/segments.csv.
"""

# Define segment benefits mapping
SEGMENT_BENEFITS = {
    'high_value': {
        'primary': ['Travel Insurance', 'Purchase Protection', 'Concierge'],
        'description': 'Premium customers with high engagement and spending'
    },
    'at_risk': {
        'primary': ['Fee Reversal', 'Cashback', 'Purchase Protection'],
        'description': 'Previously active customers showing declining engagement'
    },
    'new': {
        'primary': ['Cashback', 'Welcome Bonus', 'Purchase Protection'],
        'description': 'New customers in onboarding phase'
    },
    'dormant': {
        'primary': ['Reactivation Offer', 'Cashback', 'Fee Reversal'],
        'description': 'Inactive customers needing re-engagement'
    }
}


def assign_segment(customer_features):
    """
    Assign a customer to a segment based on RFM and behavioral features.
    
    Args:
        customer_features (dict): Dictionary with keys like:
            - recency_days: days since last transaction
            - frequency: number of transactions
            - monetary: total spending
            - days_as_customer: tenure
            
    Returns:
        str: segment name ('high_value', 'at_risk', 'new', or 'dormant')
    """
    
    # Extract features with sensible defaults
    recency = customer_features.get('recency_days', 180)
    frequency = customer_features.get('frequency', 10)
    monetary = customer_features.get('monetary', 30000)
    tenure = customer_features.get('days_as_customer', 365)
    
    # NEW CUSTOMERS (tenure < 90 days)
    if tenure < 90:
        return 'new'
    
    # HIGH VALUE (recent, frequent, high spend)
    if recency < 30 and frequency >= 20 and monetary >= 50000:
        return 'high_value'
    
    # AT RISK (was active but declining - high tenure but recently inactive)
    if tenure > 180 and recency > 60 and frequency >= 15:
        return 'at_risk'
    
    # DORMANT (inactive for a long time)
    if recency > 120 or frequency < 5:
        return 'dormant'
    
    # DEFAULT: treat as at_risk (safer fallback)
    return 'at_risk'


def get_segment_description(segment):
    """Get human-readable description of a segment."""
    return SEGMENT_BENEFITS.get(segment, {}).get('description', 'Unknown segment')


def get_segment_benefits(segment):
    """Get primary benefits for a segment."""
    return SEGMENT_BENEFITS.get(segment, {}).get('primary', ['Purchase Protection'])