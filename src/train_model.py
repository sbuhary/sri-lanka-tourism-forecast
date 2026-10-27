"""Train the Extra Trees model used by the app and save its settings."""

from pathlib import Path
import json
import hashlib
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor

folder = Path(__file__).resolve().parents[1]
csv_file = folder / 'data' / 'Tourism_Arrivals.csv'
features = ['Country', 'lag_1', 'lag_2', 'lag_3', 'lag_12', 'month_number']


def build_pipeline(model):
    preprocessing = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         ['Country', 'month_number']),
        ('history', StandardScaler(), ['lag_1', 'lag_2', 'lag_3', 'lag_12'])])
    return make_pipeline(preprocessing, model)


def latest_inputs(data, target_month, country='India'):
    inputs = {'Country': country}
    for lag in [1, 2, 3, 12]:
        month = str(target_month - lag)
        row = data.loc[data.Country.eq(country) & data.Year_Month.eq(month)]
        if len(row) != 1:
            raise ValueError(f'Missing or duplicate {country} arrivals for {month}')
        inputs[f'lag_{lag}'] = int(row.Arrivals.iloc[0])
    inputs['month_number'] = target_month.month
    return inputs


def train_model():
    data = pd.read_csv(csv_file)
    train = data.loc[data.Year_Month.ge('2019-01')]
    if train.empty or train[features + ['Arrivals']].isna().any().any():
        raise ValueError('Training data is empty or has missing inputs')
    latest_month = pd.Period(train.Year_Month.max(), freq='M')
    target_month = latest_month + 1
    default_inputs = latest_inputs(data, target_month)

    evaluation = json.loads((folder / 'models' / 'evaluation_results.json').read_text())
    model = build_pipeline(ExtraTreesRegressor(**evaluation['settings'], n_jobs=4))
    model.fit(train[features], train.Arrivals)

    prediction = float(model.predict(pd.DataFrame([default_inputs])[features])[0])

    joblib.dump(model, folder / 'models' / 'deployment_model.joblib')
    metadata = {
        'model_name': 'Extra Trees', 'settings': evaluation['settings'],
        'training_period': f'{train.Year_Month.min()} to {latest_month}', 'training_rows': len(train),
        'features': features, 'countries': sorted(train.Country.unique().tolist()),
        'sklearn_version': sklearn.__version__,
        'csv_sha256': hashlib.sha256(csv_file.read_bytes()).hexdigest(),
        'deployment_default_prediction': prediction,
        'default_inputs': default_inputs, 'default_target': str(target_month),
        'model_role': 'deployment refit; not independently evaluated',
        'evaluation': evaluation['evaluation_scope'],
        'validation_metrics': evaluation['validation']['Extra Trees'],
        'retrospective_test_metrics': evaluation['assessment']['Extra Trees'],
        'retrospective_test_baselines': {name: values for name, values in evaluation['assessment'].items() if name != 'Extra Trees'},
        'development_mae': evaluation['validation']['Extra Trees']['MAE'],
        'development_r2': evaluation['validation']['Extra Trees']['R2'],
        'development_wape_pct': evaluation['validation']['Extra Trees']['WAPE_pct']}
    (folder / 'models' / 'model_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved deployment model ({len(train)} rows). India {target_month.strftime("%B %Y")} forecast: {prediction:,.2f}')


if __name__ == '__main__':
    train_model()
