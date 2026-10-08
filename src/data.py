import pandas as pd

from src.config import DATA_PATH


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data = data[data['department_id'].notna()].sort_values(by=['department_id', 'year']).copy()

    known = data[['devs_fired', 'current_staff_count']].dropna()
    fire_rate = (known['devs_fired'] / known['current_staff_count']).median()

    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].fillna(
        data.groupby('department_id')[num_cols].transform('median')
    )

    no_history = data['devs_fired'].isna()
    data.loc[no_history, 'devs_fired'] = (
            data.loc[no_history, 'current_staff_count'] * fire_rate
    ).round()

    for dept_id in [1000164.0, 1000337.0]:
        idx = (data['department_id'] == dept_id) & (data['year'] == 2020.0)
        median_val = data.loc[data['department_id'] == dept_id, 'inbound_applications'].median()
        data.loc[idx, 'inbound_applications'] = median_val

    data['turnover_rate'] = (data['devs_fired'] + data['devs_hired']) / data['current_staff_count']

    return data

def build_features(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data = data.sort_values(by=['department_id', 'year'])

    data['yearly_cv_processed_lag1'] = data.groupby('department_id')['yearly_cv_processed'].shift(1)

    data['active_projects_count_lag1'] = data.groupby('department_id')['active_projects_count'].shift(1)
    data['active_projects_delta1'] = data.groupby('department_id')['active_projects_count'].diff(1)

    data['net_hiring'] = data['devs_hired'] - data['devs_fired']

    return data

