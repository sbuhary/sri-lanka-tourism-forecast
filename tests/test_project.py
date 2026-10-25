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
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, f"{self.info['deployment_default_prediction']:,.0f}")
        self.assertIn(pd.Period(self.info['default_target'], freq='M').strftime('%B %Y'), app.metric[0].label)

        next(field for field in app.number_input if field.label == '1 month earlier').set_value(0)
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        changed = dict(self.info['default_inputs'], lag_1=0)
        prediction = float(self.model.predict(pd.DataFrame([changed])[self.info['features']])[0])
        self.assertEqual(app.metric[0].value, f'{max(0, prediction):,.0f}')


if __name__ == '__main__':
    unittest.main()
