from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from train_model import build_pipeline, features

folder = Path(__file__).resolve().parent


def calculate_scores(actual, prediction):
    return {'MAE': mean_absolute_error(actual, prediction),
            'RMSE': root_mean_squared_error(actual, prediction),
            'R2': r2_score(actual, prediction),
            'WAPE_pct': 100 * np.abs(np.asarray(actual) - prediction).sum() / np.sum(actual)}


def track_experiments():
    import dagshub
    import mlflow

    dagshub.init(repo_owner='sbuhary', repo_name='sri-lanka-tourism-forecast', mlflow=True)
    mlflow.set_experiment('Tourism arrivals')
    data = pd.read_csv(folder / 'Tourism_Arrivals.csv')
    train = data.loc[data.Year_Month.between('2019-01', '2023-12')]
    validation = data.loc[data.Year_Month.between('2024-01', '2024-12')]
    validation = validation.loc[validation.Country.isin(train.Country.unique())]
    digest = hashlib.sha256((folder / 'Tourism_Arrivals.csv').read_bytes()).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=folder, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=folder, text=True).strip(), 'Commit changes before tracking a run.'
    tags = {'data_sha256': digest, 'source_commit': commit, 'mlflow.source.git.commit': commit,
            'evaluation_scope': 'development validation; not an independent test'}

    forest = build_pipeline(RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=4))
    boosting = build_pipeline(HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=15, l2_regularization=10, random_state=42))
    models = {
        'Ridge': build_pipeline(Ridge(alpha=10)),
        'Random Forest': forest,
        'Histogram boosting': boosting,
        'Extra Trees': build_pipeline(ExtraTreesRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=4)),
        'Voting ensemble': VotingRegressor([
            ('forest', clone(forest)), ('boosting', clone(boosting))], weights=[1, 1])}
    run_records = []
    names = list(models) + ['Previous month', 'Same month last year']
    for name in names:
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({**tags, 'role': 'validation', 'model_name': name})
            mlflow.log_params({'train_period': '2019-01 to 2023-12',
                               'validation_period': '2024-01 to 2024-12',
                               'train_rows': len(train), 'validation_rows': len(validation),
                               'features': ','.join(features), 'selection_metric': 'MAE'})
            if name in models:
                model = models[name]
                model.fit(train[features], train.Arrivals)
                prediction = np.maximum(0, model.predict(validation[features]))
                estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
                params = {key: value for key, value in estimator.get_params(deep=False).items()
                          if isinstance(value, (str, int, float, bool)) or value is None}
                mlflow.log_params(params)
                if name == 'Voting ensemble':
                    mlflow.log_params({'forest_weight': 1, 'boosting_weight': 1})
            else:
                column = 'lag_1' if name == 'Previous month' else 'lag_12'
                prediction = validation[column].to_numpy()
            scores = calculate_scores(validation.Arrivals, prediction)
            mlflow.log_metrics(scores)
            mlflow.log_dict(scores, 'validation_metrics.json')
            if name == 'Random Forest':
                # This artifact belongs to the validation fit, not the deployed model.
                mlflow.log_dict({'training_period': '2019-01 to 2023-12',
                                 'selected_settings': {'n_estimators': 150, 'max_depth': 12,
                                                       'min_samples_leaf': 5, 'random_state': 42}},
                                'selected_settings.json')
            run_records.append({'name': name, 'role': 'validation', 'run_id': run.info.run_id,
                                'metrics': scores})
            print(f'{name}: MAE {scores["MAE"]:.3f}')

    # The deployment refit is recorded without assigning validation scores to it.
    info = json.loads((folder / 'model_info.json').read_text())
    assert info['csv_sha256'] == digest
    model = joblib.load(folder / 'deployment_model.joblib')
    example = pd.DataFrame([info['default_inputs']])[features]
    assert abs(float(model.predict(example)[0]) - info['deployment_default_prediction']) < .001
    with mlflow.start_run(run_name='Random Forest - deployment') as run:
        mlflow.set_tags({**tags, 'role': 'deployment', 'evaluation_scope': 'not separately evaluated'})
        mlflow.log_params({'train_period': info['training_period'], 'train_rows': info['training_rows'],
                           'n_estimators': 150, 'max_depth': 12, 'min_samples_leaf': 5,
                           'random_state': 42, 'features': ','.join(features)})
        mlflow.log_artifact(str(folder / 'deployment_model.joblib'), artifact_path='model')
        mlflow.log_artifact(str(folder / 'model_info.json'), artifact_path='model')
        run_records.append({'name': 'Random Forest - deployment', 'role': 'deployment',
                            'run_id': run.info.run_id})
    (folder / 'tracking_results.json').write_text(json.dumps(
        {'source_commit': commit, 'data_sha256': digest, 'runs': run_records}, indent=2))
    print('Runs and deployment artifacts saved to DagsHub.')


if __name__ == '__main__':
    track_experiments()
