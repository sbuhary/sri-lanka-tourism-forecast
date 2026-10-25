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
        self.assertEqual(app.metric[0].value, f"{self.info['deployment_default_prediction']:,.0f}")
        default_target = pd.Period(self.info['default_target'], freq='M')
        self.assertEqual(app.selectbox(key='picker_year').value, default_target.year)
        self.assertEqual(app.selectbox(key='picker_month').value, default_target.month)

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

        app.button(key='next_month').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(app.info)
        app.button(key='edit_1').click().run(timeout=60)
        self.assertIsNone(app.number_input(key='draft_1').value)
        app.button(key='save_1').click().run(timeout=60)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)
        app.number_input(key='draft_1').set_value(0)
        app.button(key='save_1').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        app.button(key='reset_1').click().run(timeout=60)
        self.assertEqual(len(app.metric), 0)
        app.button(key='previous_month').click().run(timeout=60)
        self.assertIn('Germany', app.metric[0].label)

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

        app.selectbox(key='picker_year').select(target.year - 1)
        app.selectbox(key='picker_month').select(target.month)
        app.button(key='go_to_month').click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['target_month'], str(target - 12))
        self.assertIn((target - 12).strftime('%B %Y'), app.metric[0].label)



if __name__ == '__main__':
    unittest.main()
