"""Check source-table reading, country names and historical feature preparation."""

from io import BytesIO
from pathlib import Path
import sys
import unittest
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from update_data import (arrival_count, build_features, canonical_country,
                         describe_changes, discover_files, parse_workbook)
from train_model import latest_inputs
from forecast_inputs import historical_arrivals


class DataUpdateChecks(unittest.TestCase):
    def test_missing_and_zero_counts(self):
        self.assertEqual(arrival_count(0), 0)
        self.assertIsNone(arrival_count(None))
        self.assertIsNone(arrival_count('-'))
        self.assertEqual(arrival_count('1,234'), 1234)
        for value in [-1, 1.5, 'NaN', 'unknown']:
            with self.assertRaises(ValueError):
                arrival_count(value)

    def test_history_lookup_missing_and_zero(self):
        data = pd.DataFrame([
            {'Country': 'India', 'Year_Month': '2026-09', 'Arrivals': 0},
            {'Country': 'Germany', 'Year_Month': '2026-09', 'Arrivals': 500},
            {'Country': 'India', 'Year_Month': '2025-10', 'Arrivals': 100}])
        counts = historical_arrivals(data, 'India', pd.Period('2026-10', freq='M'))
        self.assertEqual(counts['lag_1'], 0)
        self.assertEqual(counts['lag_12'], 100)
        self.assertIsNone(counts['lag_2'])
        self.assertIsNone(counts['lag_3'])
        with self.assertRaises(ValueError):
            historical_arrivals(pd.concat([data, data]), 'India', pd.Period('2026-10', freq='M'))

    def test_early_history_from_prepared_lags(self):
        data = pd.DataFrame([{'Country': 'India', 'Year_Month': '2019-01', 'Arrivals': 100,
                              'lag_1': 20, 'lag_2': 0, 'lag_3': 15, 'lag_12': 10}])
        counts = historical_arrivals(data, 'India', pd.Period('2019-01', freq='M'))
        self.assertEqual(counts, {'lag_3': 15, 'lag_2': 0, 'lag_1': 20, 'lag_12': 10})

    def test_country_mapping(self):
        self.assertEqual(canonical_country('UNITED STATES OF AMERICA'), 'United States')
        self.assertEqual(canonical_country('CONGO, REPUBLIC OF THE'), 'Republic of Congo')
        self.assertEqual(canonical_country('CONGO, DEMOCRATIC REPUBLIC OF THE'),
                         'Democratic Republic of Congo')

    def test_source_discovery(self):
        links = ''.join(f'<a href="/tables/{year} arrivals.xlsx">Tourist Arrivals from All Countries - {year} (excel)</a>'
                        for year in range(2018, 2028))
        found = discover_files(links)
        self.assertEqual(len(found), 10)
        self.assertIn('%20', found[2027])
        with self.assertRaises(ValueError):
            discover_files(links.replace('All Countries - 2020', 'Other table - 2020'))
        with self.assertRaises(ValueError):
            discover_files(links + '<a href="/other.xlsx">All Countries - 2026</a>')

    def test_exact_history_and_revisions(self):
        records = [{'Year_Month': str(month), 'Country': 'India', 'Arrivals': i}
                   for i, month in enumerate(pd.period_range('2024-01', '2025-04', freq='M'))]
        old = build_features(records)
        january = old.loc[old.Year_Month.eq('2025-01')].iloc[0]
        self.assertEqual(january.lag_12, 0)
        self.assertEqual(january.lag_1, 11)
        revised = [dict(row) for row in records]
        revised[12]['Arrivals'] = 999
        new = build_features(revised)
        changes = describe_changes(old, new)
        self.assertEqual(changes['revised_rows'], 4)
        self.assertEqual(changes['added_rows'], 0)
        missing = build_features([row for row in records if row['Year_Month'] != '2025-02'])
        self.assertNotIn('2025-03', missing.Year_Month.tolist())
        with self.assertRaises(ValueError):
            build_features(records + [records[0]])

    def test_next_year_default(self):
        data = pd.DataFrame([{'Year_Month': str(month), 'Country': 'India', 'Arrivals': i}
                             for i, month in enumerate(pd.period_range('2025-01', '2026-12', freq='M'))])
        inputs = latest_inputs(data, pd.Period('2027-01', freq='M'))
        self.assertEqual(inputs['month_number'], 1)
        self.assertEqual(inputs['lag_1'], 23)
        self.assertEqual(inputs['lag_12'], 12)
        with self.assertRaises(ValueError):
            latest_inputs(data.iloc[:-1], pd.Period('2027-01', freq='M'))

    def test_workbook_totals_and_exclusions(self):
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(['Rank', 'Country', 'Jan', 'Feb', 'Total'])
        for i in range(100):
            sheet.append([i + 1, f'Country {i}', 1, 2, 3])
        sheet.append([None, 'Total', 100, 200, 300])
        def content():
            stream = BytesIO()
            book.save(stream)
            return stream.getvalue()
        records, audit = parse_workbook(content(), 2027)
        self.assertEqual(len(records), 200)
        self.assertEqual(audit['monthly_totals']['2027-02'], 200)
        sheet['E2'] = 4
        sheet['E102'] = 301
        records, audit = parse_workbook(content(), 2027)
        self.assertEqual(len(records), 198)
        self.assertEqual(audit['excluded_rows'][0]['reasons'], ['row total mismatch'])
        sheet['C102'] = 101
        with self.assertRaises(ValueError):
            parse_workbook(content(), 2027)
        book.close()


if __name__ == '__main__':
    unittest.main()
