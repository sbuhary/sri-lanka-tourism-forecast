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


def train_model():
    data = pd.read_csv(csv_file)
    # I keep the selected settings and refit using the available history.
    train = data.loc[data.Year_Month.between('2019-01', '2026-09')]
    assert not train[features + ['Arrivals']].isna().any().any()

    evaluation = json.loads((folder / 'models' / 'evaluation_results.json').read_text())
    model = build_pipeline(ExtraTreesRegressor(**evaluation['settings'], n_jobs=4))
    model.fit(train[features], train.Arrivals)

    default_inputs = {'Country': 'India', 'lag_1': 51931, 'lag_2': 47253,
                      'lag_3': 44547, 'lag_12': 48113, 'month_number': 10}
    # Check the default counts against the included dataset.
    for column, month in [('lag_1', '2026-09'), ('lag_2', '2026-08'),
                          ('lag_3', '2026-07'), ('lag_12', '2025-10')]:
        row = data.loc[data.Country.eq('India') & data.Year_Month.eq(month)]
        assert len(row) == 1 and float(row.Arrivals.iloc[0]) == default_inputs[column]
    prediction = float(model.predict(pd.DataFrame([default_inputs])[features])[0])

    joblib.dump(model, folder / 'models' / 'deployment_model.joblib')
    metadata = {
        'model_name': 'Extra Trees', 'settings': evaluation['settings'],
        'training_period': '2019-01 to 2026-09', 'training_rows': len(train),
        'features': features, 'countries': sorted(train.Country.unique().tolist()),
        'sklearn_version': sklearn.__version__,
        'csv_sha256': hashlib.sha256(csv_file.read_bytes()).hexdigest(),
        'deployment_default_prediction': prediction,
        'default_inputs': default_inputs, 'default_target': '2026-10',
        'model_role': 'deployment refit; not independently evaluated',
        'evaluation': evaluation['evaluation_scope'],
        'validation_metrics': evaluation['validation']['Extra Trees'],
        'retrospective_test_metrics': evaluation['assessment']['Extra Trees'],
        'retrospective_test_baselines': {name: values for name, values in evaluation['assessment'].items() if name != 'Extra Trees'},
        'development_mae': evaluation['validation']['Extra Trees']['MAE'],
        'development_r2': evaluation['validation']['Extra Trees']['R2'],
        'development_wape_pct': evaluation['validation']['Extra Trees']['WAPE_pct']}
    (folder / 'models' / 'model_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved deployment model ({len(train)} rows). India October 2026 forecast: {prediction:,.2f}')


if __name__ == '__main__':
    train_model()
