from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / 'data' / 'df_ITs.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'model.joblib'

TARGET = 'yearly_cv_processed'

CAT_FEATURES = ['tech_stack']

NUM_FEATURES = [
    'yearly_cv_processed_lag1',
    'active_projects_count_lag1',
    'active_projects_delta1',
    'net_hiring',
    'tech_brand_rating',
]

INTERVAL_COLS = [
    'year', 'current_staff_count', 'yearly_cv_processed', 'share_seniors',
    'active_projects_count', 'inbound_applications', 'avg_dev_salary',
    'devs_hired', 'devs_fired', 'turnover_rate', 'tech_brand_rating',
    'total_salary_budget',
]

CV_FOLDS = [
    {'train_years': [2021], 'test_year': 2022},
    {'train_years': [2021, 2022], 'test_year': 2023},
    {'train_years': [2021, 2022, 2023], 'test_year': 2024},
]

RIDGE_ALPHA = 1.0
FORECAST_YEARS = [2025, 2026, 2027]

INBOUND_OUTLIER_FACTOR = 5