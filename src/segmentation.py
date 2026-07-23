"""
User Segmentation Logic

Categorizes users into actionable segments based on RFM-style features.
Order of checks matters: high_value first, then at_risk, new, dormant, fallback.
"""


def assign_segment(features_dict):
    """
    Assign user to segment based on features.

    Segments:
    1. High-Value: Top spenders, frequent users, recently active
    2. At-Risk: Was active, now going quiet
    3. New: Recently acquired, still building history
    4. Dormant: Inactive 90+ days
    """

    recency = features_dict.get('recency_days', 999)
    frequency = features_dict.get('frequency', 0)
    monetary = features_dict.get('monetary', 0)
    age_days = features_dict.get('age_days', 0)

    if monetary > 80000 and frequency > 40 and recency < 30:
        return 'high_value'

    elif recency > 60 and frequency > 30:
        return 'at_risk'

    elif age_days < 90 or frequency < 20:
        return 'new'

    elif recency > 90:
        return 'dormant'

    return 'at_risk'


SEGMENT_BENEFITS = {
    'high_value': {
        'primary': ['Travel Insurance', 'Purchase Protection', 'Concierge'],
        'description': 'Top-tier customer: premium protection + concierge'
    },
    'at_risk': {
        'primary': ['Fee Reversal', 'Bonus Acceleration'],
        'description': 'Re-engage with high-value offers'
    },
    'new': {
        'primary': ['Welcome Bonus', 'Travel Insurance'],
        'description': 'New cardholders: onboarding benefits'
    },
    'dormant': {
        'primary': ['Reactivation Offer', 'Concierge'],
        'description': 'Dormant users: compelling re-engagement'
    }
}


if __name__ == '__main__':
    test_features = {
        'recency_days': 15,
        'frequency': 60,
        'monetary': 120000,
        'age_days': 300
    }
    segment = assign_segment(test_features)
    print(f"Test user segment: {segment}")
    print(f"Recommended benefits: {SEGMENT_BENEFITS[segment]}")
