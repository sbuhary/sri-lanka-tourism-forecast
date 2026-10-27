"""Check the dataset, saved model, forecast rules and app controls."""

from pathlib import Path
import hashlib
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

folder = Path(__file__).resolve().parents[1]


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(folder / 'data' / 'Tourism_Arrivals.csv')
        cls.info = json.loads((folder / 'models' / 'model_info.json').read_text())
        cls.model = joblib.load(folder / 'models' / 'deployment_model.joblib')

    def test_training_data(self):
        data = self.data
        self.assertEqual(len(data), self.info['training_rows'])
        self.assertFalse(data.isna().any().any())
        self.assertFalse(data.duplicated(['Country', 'Year_Month']).any())
        self.assertTrue(data.month_number.between(1, 12).all())
        numeric = data.select_dtypes(include='number')
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())
        self.assertTrue(numeric.ge(0).all().all())
        months = pd.PeriodIndex(data.Year_Month, freq='M')
        np.testing.assert_array_equal(months.month, data.month_number)
        self.assertEqual(str(months.min()), self.info['training_period'].split(' to ')[0])
        self.assertEqual(str(months.max()), self.info['training_period'].split(' to ')[1])
        self.assertEqual(str(months.max() + 1), self.info['default_target'])
        digest = hashlib.sha256((folder / 'data' / 'Tourism_Arrivals.csv').read_bytes()).hexdigest()
        self.assertEqual(digest, self.info['csv_sha256'])

    def test_saved_prediction(self):
        from sklearn.ensemble import ExtraTreesRegressor
        self.assertIsInstance(self.model.steps[-1][1], ExtraTreesRegressor)
        self.assertFalse(self.model.steps[-1][1].bootstrap)
        inputs = pd.DataFrame([self.info['default_inputs']])[self.info['features']]
        prediction = float(self.model.predict(inputs)[0])
        if self.info['csv_sha256'] == 'b1886d1337906be587388ade8a4527f649c7e8b931d78624df62facab56379e9':
            self.assertAlmostEqual(prediction, 44498.88, delta=0.05)
        self.assertAlmostEqual(prediction, self.info['deployment_default_prediction'], places=6)
        inputs.loc[:, ['lag_1', 'lag_2', 'lag_3', 'lag_12']] = 0
        self.assertTrue(np.isfinite(self.model.predict(inputs)).all())
        self.assertGreaterEqual(float(self.model.predict(inputs)[0]), 0)

    def test_app(self):
        app = AppTest.from_file(str(folder / 'app.py')).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.table), 2)
        self.assertEqual(len(app.table[0].value), 6)
        self.assertIn('Extra Trees', app.table[0].value.index)
        self.assertIn('Same month last year', app.table[1].value.index)
        self.assertEqual(len(app.dataframe[0].value), 200)
        self.assertEqual([tab.label for tab in app.tabs], ['Forecast', 'Data', 'Model'])
        app.selectbox(key='data_market').set_value('Germany').run(timeout=60)
        self.assertTrue(app.dataframe[0].value.Country.eq('Germany').all())
        app.selectbox(key='data_market').set_value('All markets').run(timeout=60)
        self.assertEqual(app.metric[0].value, f"{self.info['deployment_default_prediction']:,.0f}")
        from src.forecast_inputs import forecast_outlook
        expected, _ = forecast_outlook(self.model, self.info['features'], self.data, self.info['default_inputs']['Country'], self.info['default_target'])
        self.assertEqual([metric.value for metric in app.metric], [f'{row["Estimated arrivals"]:,.0f}' for row in expected])
        app.radio(key='horizon').set_value(1).run(timeout=60)
        self.assertEqual(len(app.metric), 1)
        app.radio(key='horizon').set_value(2).run(timeout=60)
        self.assertEqual(len(app.metric), 2)
        self.assertEqual([metric.value for metric in app.metric], [f'{row["Estimated arrivals"]:,.0f}' for row in expected[:2]])
        app.radio(key='horizon').set_value(3).run(timeout=60)

        app.button(key='pick_Germany').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='market').value, 'Germany')
        self.assertIn('Germany', app.metric[0].label)
        target = pd.Period(self.info['default_target'], freq='M')
        inputs = {'Country': 'Germany', 'month_number': target.month}
        for lag in [1, 2, 3, 12]:
            row = self.data.loc[self.data.Country.eq('Germany') & self.data.Year_Month.eq(str(target - lag))]
            inputs[f'lag_{lag}'] = int(row.Arrivals.iloc[0])
        prediction = float(self.model.predict(pd.DataFrame([inputs])[self.info['features']])[0])
        self.assertEqual(app.metric[0].value, f'{prediction:,.0f}')

        app.button(key='pick_India').click().run(timeout=60)
        app.button(key='edit_1').click().run(timeout=60)
        app.number_input(key='draft_1').set_value(0)
        app.button(key='cancel_1').click().run(timeout=60)
        self.assertEqual(app.metric[0].value, f"{self.info['deployment_default_prediction']:,.0f}")

        app.button(key='edit_1').click().run(timeout=60)
        app.number_input(key='draft_1').set_value(0)
        app.button(key='save_1').click().run(timeout=60)
        self.assertFalse(app.exception)
        changed = dict(self.info['default_inputs'], lag_1=0)
        prediction = float(self.model.predict(pd.DataFrame([changed])[self.info['features']])[0])
        self.assertEqual(app.metric[0].value, f'{max(0, prediction):,.0f}')
        app.button(key='reset_1').click().run(timeout=60)
        self.assertEqual(app.metric[0].value, f"{self.info['deployment_default_prediction']:,.0f}")

        for lag in [1, 2]:
            app.button(key=f'edit_{lag}').click().run(timeout=60)
            app.number_input(key=f'draft_{lag}').set_value(0)
            app.button(key=f'save_{lag}').click().run(timeout=60)
        app.button(key='reset_all').click().run(timeout=60)
        self.assertEqual(app.metric[0].value, f"{self.info['deployment_default_prediction']:,.0f}")



    def test_recursive_outlook(self):
        from src.forecast_inputs import forecast_outlook

        class NextCount:
            def predict(self, inputs):
                return inputs.lag_1.to_numpy() + 1

        target = pd.Period(self.info['default_target'], freq='M')
        country = self.info['default_inputs']['Country']
        rows, missing = forecast_outlook(NextCount(), self.info['features'], self.data, country, target)
        expected = self.info['default_inputs']['lag_1']
        self.assertFalse(missing)
        self.assertEqual([row['Estimated arrivals'] for row in rows], [expected + 1, expected + 2, expected + 3])
        future = self.data.iloc[[0]].copy()
        future['Country'], future['Year_Month'], future['Arrivals'] = country, str(target), 99999999
        changed, _ = forecast_outlook(NextCount(), self.info['features'], pd.concat([self.data, future]), country, target)
        self.assertEqual(rows, changed)
        without_last = self.data.loc[~(self.data.Country.eq(country) & self.data.Year_Month.eq(str(target - 1)))]
        rows, missing = forecast_outlook(NextCount(), self.info['features'], without_last, country, target)
        self.assertFalse(rows)
        self.assertIn(str(target - 1), missing)
        rows, missing = forecast_outlook(NextCount(), self.info['features'], without_last, country, target, overrides={'lag_1': 0})
        self.assertEqual(rows[0]['Estimated arrivals'], 1)




if __name__ == '__main__':
    unittest.main()
