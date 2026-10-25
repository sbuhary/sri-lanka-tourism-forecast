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


def forecast_outlook(model, features, data, country, start, months=3, overrides=None):
    """Forecast consecutively, reusing estimates only within this outlook."""
    if months not in (1, 2, 3):
        raise ValueError('Choose between one and three months')
    start = pd.Period(start, freq='M')
    supplied = {str(start - int(field.split('_')[1])): value
                for field, value in (overrides or {}).items()}
    estimates, rows = {}, []
    for step in range(months):
        target = start + step
        counts = historical_arrivals(data.loc[data.Year_Month.lt(str(start))], country, target)
        for lag in lags:
            month = str(target - lag)
            # Never use later actuals when checking a historical forecast origin.
            if month >= str(start):
                counts[f'lag_{lag}'] = estimates.get(month)
            elif month in supplied:
                counts[f'lag_{lag}'] = supplied[month]
        missing = [str(target - lag) for lag in lags if counts[f'lag_{lag}'] is None]
        if missing:
            return rows, missing
        inputs = {'Country': country, **counts, 'month_number': target.month}
        value = max(0, float(model.predict(pd.DataFrame([inputs])[features])[0]))
        estimates[str(target)] = value
        rows.append({'Month': str(target), 'Estimated arrivals': value})
    return rows, []
