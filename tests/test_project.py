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
        cls.data = pd.read_csv(folder / 'Tourism_Arrivals.csv')
        cls.info = json.loads((folder / 'model_info.json').read_text())
        cls.model = joblib.load(folder / 'deployment_model.joblib')

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
        self.assertEqual(str(months.min()), '2019-01')
        self.assertEqual(str(months.max()), '2026-09')
        digest = hashlib.sha256((folder / 'Tourism_Arrivals.csv').read_bytes()).hexdigest()
        self.assertEqual(digest, self.info['csv_sha256'])

    def test_saved_prediction(self):
        inputs = pd.DataFrame([self.info['default_inputs']])[self.info['features']]
        prediction = float(self.model.predict(inputs)[0])
        self.assertAlmostEqual(prediction, 50521.55, delta=0.05)
        self.assertAlmostEqual(prediction, self.info['deployment_default_prediction'], places=6)
        inputs.loc[:, ['lag_1', 'lag_2', 'lag_3', 'lag_12']] = 0
        self.assertTrue(np.isfinite(self.model.predict(inputs)).all())
        self.assertGreaterEqual(float(self.model.predict(inputs)[0]), 0)

    def test_app(self):
        app = AppTest.from_file(str(folder / 'app.py')).run(timeout=60)
        self.assertFalse(app.exception)
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '50,522')
        self.assertIn('October 2026', app.metric[0].label)


if __name__ == '__main__':
    unittest.main()
