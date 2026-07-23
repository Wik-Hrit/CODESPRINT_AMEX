import pandas as pd
import numpy as np


def create_features(master_data):
    """
    Transform raw transaction data into ML features.
    Input: master_data.csv (transactions x demographic columns)
    Output: features DataFrame (n_users x 25 features)

    Features:
    1. Recency: days since last transaction (relative to dataset's max Time)
    2. Frequency: transaction count
    3. Monetary: total spend
    4. Category diversity: # of distinct merchant-category buckets
       (simulated by binning Amount into deciles as a category proxy,
       since the real dataset has no merchant-category field)
    5-16. Monthly spend (12 buckets, derived from Time)
    17. Avg transaction size
    18. Std dev of transaction size
    19. Max transaction
    20. Transaction trend (later-half spend vs earlier-half spend)
    21-25. Demographics: age, income, card_type (one-hot: gold, platinum), age_days
    """

    df = master_data.copy()

    # Merchant-category proxy: bin Amount into 10 buckets per user's spend range
    df['category_bucket'] = pd.qcut(df['Amount'], q=10, labels=False, duplicates='drop')

    # Convert Time (seconds since dataset start) into a day index and a month bucket
    df['day_index'] = (df['Time'] // 86400).astype(int)
    max_day = df['day_index'].max()
    df['month_bucket'] = pd.cut(df['day_index'], bins=12, labels=range(1, 13)).astype(int)

    user_groups = df.groupby('user_id')

    features = pd.DataFrame({'user_id': user_groups.size().index})

    # 1: Recency - days since each user's last transaction, relative to the most
    #    recent transaction in the whole dataset (stand-in for "today")
    last_day_per_user = user_groups['day_index'].max()
    features['recency_days'] = (max_day - last_day_per_user).values

    # 2-3: Frequency & Monetary
    features['frequency'] = user_groups.size().values
    features['monetary'] = user_groups['Amount'].sum().values

    # 4: Category diversity
    features['category_diversity'] = user_groups['category_bucket'].nunique().values

    # 5-16: Monthly spend
    monthly = df.pivot_table(index='user_id', columns='month_bucket', values='Amount',
                              aggfunc='sum', fill_value=0)
    for month in range(1, 13):
        col = monthly[month] if month in monthly.columns else pd.Series(0, index=monthly.index)
        features[f'monthly_spend_{month}'] = features['user_id'].map(col).fillna(0).values

    # 17-19: Transaction size stats
    features['avg_transaction'] = user_groups['Amount'].mean().values
    features['std_transaction'] = user_groups['Amount'].std().fillna(0).values
    features['max_transaction'] = user_groups['Amount'].max().values

    # 20: Trend - spend in the later half of the user's history vs the earlier half
    def trend(group):
        mid = group['day_index'].median()
        early = group.loc[group['day_index'] <= mid, 'Amount'].sum()
        late = group.loc[group['day_index'] > mid, 'Amount'].sum()
        return 0.0 if (early + late) == 0 else (late - early) / (early + late)

    features['transaction_trend'] = user_groups.apply(trend).values

    # 21-25: Demographics
    demo = df[['user_id', 'age', 'income_proxy', 'card_type']].drop_duplicates(subset=['user_id']).copy()
    demo['card_type_gold'] = (demo['card_type'] == 'Gold').astype(int)
    demo['card_type_platinum'] = (demo['card_type'] == 'Platinum').astype(int)

    features = features.merge(
        demo[['user_id', 'age', 'income_proxy', 'card_type_gold', 'card_type_platinum']],
        on='user_id', how='left'
    )

    # age_days: how long this user's transaction history spans (used by segmentation)
    first_day_per_user = user_groups['day_index'].min()
    features['age_days'] = (last_day_per_user - first_day_per_user).values

    features = features.fillna(0)

    print(f"Created {len(features)} users x {len(features.columns)} features")

    return features


if __name__ == '__main__':
    master_data = pd.read_csv('data/master_data.csv')
    features_df = create_features(master_data)
    features_df.to_csv('data/features.csv', index=False)
    print("Features saved to data/features.csv")
    print(features_df.head())
