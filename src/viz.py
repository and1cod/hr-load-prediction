import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

PRIMARY   = '#3a7ca5'
ABOVE_AVG = '#c44e52'
BELOW_AVG = '#55a868'
MEAN_C    = 'crimson'
MEDIAN_C  = 'seagreen'
PALETTE   = 'crest'


def visualize_categorical(data: pd.DataFrame, column: str):
    s = data[column]
    n = len(s)
    counts = s.value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(9, max(2.2, 0.5 * len(counts) + 1.2)))
    bars = ax.barh(counts.index.astype(str), counts.values,
                    color=sns.color_palette(PALETTE, len(counts)))
    ax.bar_label(bars, labels=[f'{v:,} · {v / n * 100:.1f}%' for v in counts.values],
                 padding=4, fontsize=9)

    ax.margins(x=0.18)
    ax.set_title(f'{column} — распределение (n={n:,}, уникальных={s.nunique()})',
                 fontsize=12, weight='bold')
    ax.set_xlabel('Количество департаментов')
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()


def visualize_numeric(data: pd.DataFrame, column: str, bins: int = 40):
    s = data[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={'width_ratios': [3, 1]})

    sns.histplot(s, bins=bins, kde=True, ax=axes[0],
                 color=PRIMARY, edgecolor='white', linewidth=0.3)
    axes[0].axvline(s.mean(), color=MEAN_C, ls='--', lw=1.6, label=f'среднее = {s.mean():.1f}')
    axes[0].axvline(s.median(), color=MEDIAN_C, ls='--', lw=1.6, label=f'медиана = {s.median():.1f}')
    axes[0].legend(frameon=False)
    axes[0].set_title(f'{column} — распределение', fontsize=12, weight='bold')
    axes[0].set_xlabel(column)

    ax = axes[1]
    sns.boxplot(y=s, ax=ax, color=PRIMARY, width=0.5)
    ax.set_title('Ящик с усами', fontsize=11)
    ax.set_ylabel('')
    ax.set_xticks([])

    q1, med, q3 = s.quantile([.25, .5, .75])
    iqr = q3 - q1
    lo = s[s >= q1 - 1.5 * iqr].min()
    hi = s[s <= q3 + 1.5 * iqr].max()
    summary = {'мин. ус': lo, 'Q1': q1, 'медиана': med, 'Q3': q3, 'макс. ус': hi}

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(list(summary.values()))
    ax2.set_yticklabels([f'{k}: {v:.1f}' for k, v in summary.items()], fontsize=9)
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    sns.despine(ax=axes[0])
    plt.tight_layout()
    plt.show()


def visualize_categorical_overview(data: pd.DataFrame, exclude_columns=None):
    exclude_columns = exclude_columns or []
    for column in data.columns:
        if column in exclude_columns:
            continue
        if not pd.api.types.is_numeric_dtype(data[column]):
            visualize_categorical(data, column)


def visualize_numeric_overview(data: pd.DataFrame, exclude_columns=None):
    exclude_columns = exclude_columns or []
    for column in data.columns:
        if column in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            visualize_numeric(data, column)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = 'Матрица корреляции'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap=PALETTE, fmt='.2f')
    plt.title(title, fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_forecast(hist_df: pd.DataFrame, forecast_df: pd.DataFrame,
                   target: str = 'yearly_cv_processed'):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(hist_df['year'], hist_df[target],
            marker='o', label='Исторические данные', color=PRIMARY)
    ax.plot(forecast_df['year'], forecast_df[target],
            marker='o', label='Прогноз', color=ABOVE_AVG, linestyle='--')

    bridge = pd.DataFrame({
        'year': [hist_df.iloc[-1]['year'], forecast_df.iloc[0]['year']],
        target: [hist_df.iloc[-1][target], forecast_df.iloc[0][target]],
    })
    ax.plot(bridge['year'], bridge[target], color=ABOVE_AVG, linestyle='--', alpha=0.5)

    split_x = (hist_df.iloc[-1]['year'] + forecast_df.iloc[0]['year']) / 2
    ax.axvline(x=split_x, color='#888', linestyle=':', alpha=0.7, label='→ прогноз')

    ax.set_title('Прогноз нагрузки на рекрутинг 2025-2027', fontsize=13, weight='bold')
    ax.set_ylabel(target)
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.show()
