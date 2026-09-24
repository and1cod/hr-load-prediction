import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CAT_FEATURES, NUM_FEATURES, TARGET, RIDGE_ALPHA


def build_design_matrix(train: pd.DataFrame, test: pd.DataFrame,
                         cat_features=CAT_FEATURES, num_features=NUM_FEATURES):

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    train_cat = ohe.fit_transform(train[cat_features])
    test_cat = ohe.transform(test[cat_features])

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[num_features])
    test_scaled = scaler.transform(test[num_features])

    scaler_cols = scaler.get_feature_names_out()
    ohe_cols = ohe.get_feature_names_out(cat_features)

    X_train = pd.concat([
        pd.DataFrame(train_scaled, columns=scaler_cols, index=train.index),
        pd.DataFrame(train_cat, columns=ohe_cols, index=train.index),
    ], axis=1)
    X_test = pd.concat([
        pd.DataFrame(test_scaled, columns=scaler_cols, index=test.index),
        pd.DataFrame(test_cat, columns=ohe_cols, index=test.index),
    ], axis=1)

    return X_train, X_test, ohe, scaler


def run_cv(data: pd.DataFrame, folds, target=TARGET,
           cat_features=CAT_FEATURES, num_features=NUM_FEATURES,
           alpha=RIDGE_ALPHA, verbose=True):

    fold_metrics = []
    model, X_train = None, None

    for fold in folds:
        train = data[data['year'].isin(fold['train_years'])].copy()
        test = data[data['year'] == fold['test_year']].copy()

        X_train, X_test, _, _ = build_design_matrix(train, test, cat_features, num_features)

        model = Ridge(alpha=alpha)
        model.fit(X_train, train[target])
        preds = model.predict(X_test)

        mae = mean_absolute_error(test[target], preds)
        r2 = r2_score(test[target], preds)
        fold_metrics.append({'test_year': fold['test_year'], 'mae': mae, 'r2': r2})

        if verbose:
            print(f'Результаты предсказания {fold["test_year"]}')
            print(f'MAE:{mae}')
            print(f'R2:{r2}')
            print()

    return fold_metrics, model, X_train


def get_coefficients(model: Ridge, feature_names) -> pd.DataFrame:
    return pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
    }).sort_values('coefficient', key=abs, ascending=False)


def train_final_model(data: pd.DataFrame, target=TARGET,
                       cat_features=CAT_FEATURES, num_features=NUM_FEATURES,
                       alpha=RIDGE_ALPHA, min_year=2020):

    train_full = data[data['year'] > min_year].copy()

    ohe = OneHotEncoder(sparse_output=False, drop='first')
    train_cat = ohe.fit_transform(train_full[cat_features])

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_full[num_features])

    scaler_cols = scaler.get_feature_names_out()
    ohe_cols = ohe.get_feature_names_out(cat_features)

    X_train_full = pd.concat([
        pd.DataFrame(train_scaled, columns=scaler_cols, index=train_full.index),
        pd.DataFrame(train_cat, columns=ohe_cols, index=train_full.index),
    ], axis=1)

    model = Ridge(alpha=alpha)
    model.fit(X_train_full, train_full[target])

    return model, ohe, scaler


def forecast_future(data: pd.DataFrame, model: Ridge, ohe: OneHotEncoder, scaler: StandardScaler,
                     years, base_year=2024, target=TARGET,
                     cat_features=CAT_FEATURES, num_features=NUM_FEATURES):

    scaler_cols = scaler.get_feature_names_out()
    ohe_cols = ohe.get_feature_names_out(cat_features)

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

        future_scaled = scaler.transform(future[num_features])
        future_scaled_df = pd.DataFrame(future_scaled, columns=scaler_cols, index=future.index)

        future_ohe = ohe.transform(future[cat_features])
        future_ohe_df = pd.DataFrame(future_ohe, columns=ohe_cols, index=future.index)

        X_future = pd.concat([future_scaled_df, future_ohe_df], axis=1)
        preds = model.predict(X_future)
        future[target] = preds

        results[future_year] = preds
        current = future.copy()

    return results
