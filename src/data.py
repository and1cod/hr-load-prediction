import pandas as pd

from src.config import DATA_PATH


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data = data.sort_values(by=['department_id', 'year'])
    data = data[data['department_id'].isna() == False]

    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].fillna(
        data.groupby('department_id')[num_cols].transform('median')
    )

    idx = data['department_id'] == 1000456.0
    data.loc[idx, 'devs_fired'] = (
            data.loc[idx, 'current_staff_count'].shift(1)
            + data.loc[idx, 'devs_hired']
            - data.loc[idx, 'current_staff_count']
    )
    data['devs_fired'] = data['devs_fired'].fillna(
        data.groupby('department_id')['devs_fired'].transform('median')
    )

    for dept_id in [1000164.0, 1000337.0]:
        idx = (data['department_id'] == dept_id) & (data['year'] == 2020.0)
        median_val = data[data['department_id'] == dept_id]['inbound_applications'].median()
        data.loc[idx, 'inbound_applications'] = median_val

    return data

def build_features(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data = data.sort_values(by=['department_id', 'year'])

    data['yearly_cv_processed_lag1'] = data.groupby('department_id')['yearly_cv_processed'].shift(1)

    data['active_projects_count_lag1'] = data.groupby('department_id')['active_projects_count'].shift(1)
    data['active_projects_delta1'] = data.groupby('department_id')['active_projects_count'].diff(1)

    data['net_hiring'] = data['devs_hired'] - data['devs_fired']

    return data

