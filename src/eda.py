import pandas as pd

try:
    from IPython.display import display
except ImportError:  # запуск вне Jupyter
    display = print


def overview(df: pd.DataFrame, name: str = '', keys=None, head: int = 5, stats: bool = False):

    if name:
        print(f'== {name} ==')

    line = f'Полных дублей: {df.duplicated().sum()}'
    if keys:
        line += f' | дублей по {keys}: {df.duplicated(keys).sum()}'
    print(line)

    info = pd.DataFrame({
        'тип': df.dtypes.astype(str),
        'пропусков': df.isna().sum(),
        '%': (df.isna().mean() * 100).round(1),
        'уникальных': df.nunique(),
        'пример': df.iloc[0] if len(df) else None,
    })
    display(info)

    if stats and df.select_dtypes('number').shape[1]:
        display(df.describe().T)

    display(df.head(head))


def show_department(data: pd.DataFrame, department_id: float, columns=None) -> pd.DataFrame:

    subset = data[data['department_id'] == department_id]
    if columns:
        subset = subset[columns]
    return subset
