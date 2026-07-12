"""
Core analysis functions: industry cost breakdown, attack vector trends, model training.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def industry_cost_summary(df, min_incidents=10):
    """Average breach cost by industry, filtered to industries with enough data to be reliable."""
    stats = df.groupby('industry_name')['total_loss_usd'].agg(['mean', 'count'])
    reliable = stats[stats['count'] >= min_incidents].sort_values('mean', ascending=False)
    return reliable


def attack_vector_trends(df):
    """Incident counts per attack vector per year."""
    return df.groupby(['incident_year', 'attack_vector_primary']).size().unstack(fill_value=0)


def market_reaction_summary(df):
    """Summarize stock market reaction stats for public companies with market data."""
    market_df = df.dropna(subset=['abnormal_return_1d'])
    significant_negative = market_df[
        (market_df['p_value_1d'] < 0.05) & (market_df['abnormal_return_1d'] < 0)
    ]
    summary = {
        'n_companies': len(market_df),
        'n_significant_negative': len(significant_negative),
        'pct_significant_negative': len(significant_negative) / len(market_df) * 100,
        'mean_abnormal_return': market_df['abnormal_return_1d'].mean(),
    }
    return summary, market_df


def prepare_model_features(df):
    """Build the feature matrix and target for the high-loss prediction model."""
    features = ['company_revenue_usd', 'employee_count', 'is_public_company',
                'industry_name', 'attack_vector_primary', 'data_compromised_records',
                'downtime_hours']

    model_df = df.dropna(subset=['total_loss_usd']).copy()
    median_loss = model_df['total_loss_usd'].median()
    model_df['high_loss'] = (model_df['total_loss_usd'] > median_loss).astype(int)

    model_data = model_df[features + ['high_loss']].copy()
    # Structurally missing, not randomly missing - 0 is a valid value here
    model_data['downtime_hours'] = model_data['downtime_hours'].fillna(0)
    model_data['data_compromised_records'] = model_data['data_compromised_records'].fillna(0)
    model_data = model_data.dropna()

    X = model_data[features].copy()
    y = model_data['high_loss']

    le_industry = LabelEncoder()
    le_vector = LabelEncoder()
    X['industry_name'] = le_industry.fit_transform(X['industry_name'])
    X['attack_vector_primary'] = le_vector.fit_transform(X['attack_vector_primary'])

    return X, y, median_loss, features


def train_loss_model(X, y, random_state=42):
    """Train a random forest classifier predicting high vs low loss breaches."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=8)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, report, importances