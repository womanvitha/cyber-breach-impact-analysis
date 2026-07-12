"""
Chart generation for the breach impact analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def plot_industry_costs(industry_stats, output_path='visuals/avg_loss_by_industry.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    industry_stats['mean'].sort_values(ascending=False).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Average Total Loss (USD)')
    ax.set_title('Average Breach Cost by Industry (n ≥ 10 incidents)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_attack_vector_trends(vector_by_year, vectors=('phishing', 'apt', 'ransomware', 'ddos'),
                               output_path='visuals/attack_vector_trends.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    vector_by_year[list(vectors)].plot(kind='line', marker='o', ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Incidents')
    ax.set_title('Attack Vector Trends Over Time')
    ax.legend(title='Attack Vector')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_market_reaction(market_df, output_path='visuals/market_reaction_distribution.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(market_df['abnormal_return_1d'], bins=30, color='darkred', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='No abnormal effect')
    mean_return = market_df['abnormal_return_1d'].mean()
    ax.axvline(mean_return, color='blue', linestyle='-', linewidth=2, label=f"Mean: {mean_return:.1%}")
    ax.set_xlabel('1-Day Abnormal Return')
    ax.set_ylabel('Number of Companies')
    ax.set_title('Stock Market Reaction to Breach Disclosure (1 Day After)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, output_path='visuals/feature_importance.png'):
    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Importance: Predicting High-Loss Breaches')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)