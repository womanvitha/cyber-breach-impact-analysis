"""
Cleans and merges the three raw breach datasets into a single analysis-ready file.
"""
import pandas as pd

NAICS_MAP = {
    '11': 'Agriculture/Forestry/Fishing',
    '21': 'Mining/Oil & Gas',
    '22': 'Utilities',
    '23': 'Construction',
    '31-33': 'Manufacturing',
    '42': 'Wholesale Trade',
    '44-45': 'Retail Trade',
    '48-49': 'Transportation/Warehousing',
    '51': 'Information/Tech',
    '52': 'Finance & Insurance',
    '53': 'Real Estate',
    '54': 'Professional/Scientific Services',
    '55': 'Management of Companies',
    '56': 'Administrative/Support Services',
    '61': 'Education',
    '62': 'Healthcare',
    '71': 'Arts/Entertainment/Recreation',
    '72': 'Accommodation/Food Services',
    '81': 'Other Services',
    '92': 'Public Administration/Government',

}


def load_raw_data(raw_dir='data/raw'):
    """Load the three raw CSV files."""
    incidents = pd.read_csv(f'{raw_dir}/incidents_master.csv')
    financial = pd.read_csv(f'{raw_dir}/financial_impact.csv')
    market = pd.read_csv(f'{raw_dir}/market_impact.csv')
    return incidents, financial, market


def merge_datasets(incidents, financial, market):
    """Merge all three tables on incident_id."""
    df = incidents.merge(financial, on='incident_id', how='left')
    df = df.merge(market, on='incident_id', how='left')
    return df


def clean_duplicate_columns(df):
    """Remove duplicate columns created by the merge (notes, created_at, updated_at)."""
    cols = df.columns.tolist()
    notes_positions = [i for i, c in enumerate(cols) if c == 'notes']
    created_positions = [i for i, c in enumerate(cols) if c == 'created_at']
    updated_positions = [i for i, c in enumerate(cols) if c == 'updated_at']

    drop_positions = notes_positions[1:] + created_positions + updated_positions
    df = df.drop(df.columns[drop_positions], axis=1)
    return df


def add_industry_names(df):
    """Map NAICS industry codes to readable names."""
    df['industry_name'] = df['industry_primary'].astype(str).map(NAICS_MAP)
    return df


def add_year_column(df):
    """Extract incident year from incident_date for time-based analysis."""
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    df['incident_year'] = df['incident_date'].dt.year
    return df


def clean_pipeline(raw_dir='data/raw', output_path='data/processed/merged_breach_data.csv'):
    """Run the full cleaning pipeline and save the result."""
    incidents, financial, market = load_raw_data(raw_dir)
    df = merge_datasets(incidents, financial, market)
    df = clean_duplicate_columns(df)
    df = add_industry_names(df)
    df = add_year_column(df)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path} — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


if __name__ == '__main__':
    clean_pipeline()