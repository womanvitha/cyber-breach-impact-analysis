"""
Runs the full analysis pipeline end-to-end: clean data, run analysis, generate all charts.
"""
from clean_data import clean_pipeline
from analysis import (
    industry_cost_summary,
    attack_vector_trends,
    market_reaction_summary,
    prepare_model_features,
    train_loss_model,
)
from visualize import (
    plot_industry_costs,
    plot_attack_vector_trends,
    plot_market_reaction,
    plot_feature_importance,
)

def main():
    print("Step 1: Cleaning and merging data...")
    df = clean_pipeline()

    print("\nStep 2: Industry cost analysis...")
    industry_stats = industry_cost_summary(df)
    plot_industry_costs(industry_stats)
    print(industry_stats)

    print("\nStep 3: Attack vector trends...")
    vector_trends = attack_vector_trends(df)
    plot_attack_vector_trends(vector_trends)

    print("\nStep 4: Market reaction analysis...")
    market_summary, market_df = market_reaction_summary(df)
    plot_market_reaction(market_df)
    print(market_summary)

    print("\nStep 5: Training predictive model...")
    X, y, median_loss, features = prepare_model_features(df)
    model, report, importances = train_loss_model(X, y)
    plot_feature_importance(importances)
    print(f"Median loss threshold: ${median_loss:,.0f}")
    print(f"Model accuracy: {report['accuracy']:.2f}")
    print(importances)

    print("\nAll done! Charts saved to visuals/, cleaned data saved to data/processed/")

if __name__ == '__main__':
    main()