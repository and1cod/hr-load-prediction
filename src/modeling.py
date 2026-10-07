import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import CAT_FEATURES, NUM_FEATURES, TARGET, RIDGE_ALPHA


def build_pipeline(cat_features=CAT_FEATURES, num_features=NUM_FEATURES, alpha=RIDGE_ALPHA) -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, drop='first', handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=alpha)),
    ])

def run_cv(data: pd.DataFrame, folds, target=TARGET,
           cat_features=CAT_FEATURES, num_features=NUM_FEATURES,
           alpha=RIDGE_ALPHA, verbose=True):

    features = cat_features + num_features
    fold_metrics = []

    for fold in folds:
        train = data[data['year'].isin(fold['train_years'])].copy()
        test = data[data['year'] == fold['test_year']].copy()

        pipe = build_pipeline(cat_features, num_features, alpha)

        pipe.fit(train.loc[:, features], train[target])
        preds = pipe.predict(test.loc[:, cat_features + num_features])

        mae = mean_absolute_error(test[target], preds)
        r2 = r2_score(test[target], preds)
        fold_metrics.append({'test_year': fold['test_year'], 'mae': mae, 'r2': r2})

        if verbose:
            print(f'Результаты предсказания {fold["test_year"]}')
            print(f'MAE:{mae}')
            print(f'R2:{r2}')
            print()

    return fold_metrics, pipe

def train_final_model(data: pd.DataFrame, target=TARGET,
                       cat_features=CAT_FEATURES, num_features=NUM_FEATURES,
                       alpha=RIDGE_ALPHA, min_year=2020):

    features = cat_features + num_features

    pipe = build_pipeline(cat_features, num_features, alpha)

    train_full = data[data['year'] > min_year].copy()

    pipe.fit(train_full.loc[:, features], train_full[target])

    return pipe

def get_coefficients(pipe: Pipeline) -> pd.DataFrame:
    return pd.DataFrame({
        'feature': pipe.named_steps['preprocess'].get_feature_names_out(),
        'coefficient': pipe.named_steps['model'].coef_,
    }).sort_values('coefficient', key=abs, ascending=False)

def forecast_future(data: pd.DataFrame, pipe: Pipeline,
                     years, base_year=2024, target=TARGET,
                     cat_features=CAT_FEATURES, num_features=NUM_FEATURES):

    features = cat_features + num_features
    current = data[data['year'] == base_year].copy()
    results = {}

    for future_year in years:
        future = current.copy()
        future['year'] = future_year

        future['yearly_cv_processed_lag1'] = current['yearly_cv_processed']
        future['active_projects_count_lag1'] = current['active_projects_count']
        future['active_projects_delta1'] = future['active_projects_count'] - current['active_projects_count']
        future['net_hiring'] = current['net_hiring']
        future['tech_brand_rating'] = current['tech_brand_rating']

        preds = pipe.predict(future.loc[:, features])
        future[target] = preds

        results[future_year] = preds
        current = future.copy()

    return results
