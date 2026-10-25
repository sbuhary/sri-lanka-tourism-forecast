"""Look up the published monthly counts needed by the forecast."""
import pandas as pd

lags = [12, 3, 2, 1]


def historical_arrivals(data, country, target):
    history = data.loc[data.Country.eq(country)].set_index('Year_Month').Arrivals
    if not history.index.is_unique:
        raise ValueError(f'Duplicate monthly records for {country}')
    target_row = data.loc[data.Country.eq(country) & data.Year_Month.eq(str(target))]
    counts = {}
    for lag in lags:
        month = str(target - lag)
        value = history.get(month)
        if (value is None or pd.isna(value)) and len(target_row) == 1 and f'lag_{lag}' in target_row:
            # Early source counts are also preserved in the prepared lag columns.
            value = target_row.iloc[0][f'lag_{lag}']
        counts[f'lag_{lag}'] = None if value is None or pd.isna(value) else int(value)
    return counts
