# Sri Lanka Tourist Arrivals Forecast

Predict monthly arrivals by origin market using Extra Trees.

[Azure app](https://tourism-forecast.calmisland-751fb489.eastus.azurecontainerapps.io/) · [Streamlit app](https://sri-lanka-tourism-forecast.streamlit.app/) · [Notebook](notebooks/Tourism_Arrivals_Analysis.ipynb)

## Run locally

Use Python 3.13 from the repo root:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved model is included. To train it again, run `python src/train_model.py`.

## Data

The CSV has 17,053 rows from January 2019 to September 2026, one market per month. It comes from nine [official SLTDA tables](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries). The 2018 counts supply the first annual lag. Clear country-name differences were corrected, zeros were kept, and unresolved rows or rows with missing history were removed.

The six inputs are market, calendar month and arrivals 1, 2, 3 and 12 months earlier. `Arrivals` is the target. Year labels the period but is not an input. The data contains public totals, not personal travel records.

## Model and results

Extra Trees averages 150 trees with random split thresholds. Each tree uses all training rows and encoded features. A check on 2023 selected depth 8 and minimum leaf size 10. See the [paper](https://doi.org/10.1007/s10994-006-6226-1) and [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

After training on 2019–2023, the 2024 results were:

| Trained model | MAE | R² |
| --- | ---: | ---: |
| Random Forest + CatBoost + LightGBM vote | 227.10 | 0.934 |
| CatBoost | 232.30 | 0.941 |
| Tuned Random Forest | 235.84 | 0.931 |
| Extra Trees | 258.39 | 0.904 |

MAE is average error in arrivals; lower is better. R² is not percentage accuracy. Extra Trees is the main algorithm studied, but other models have lower error.

Two simple rules provide a check: repeat the previous month's count, or repeat the same month last year. These **baselines do not train or learn**. They help us judge whether ML adds value. The yearly rule beats Extra Trees in the later period and at longer lead times.

The notebook contains tuning, all comparisons, market errors, three-month checks, SHAP and LIME. Months stay in time order. The later period was already examined, so it is not an independent final test.

## App

**Forecast:** choose a market and one to three months ahead. Starting counts load from the dataset and can be edited or reset. Later forecasts reuse earlier predictions. Missing required history stops the forecast.

**Data:** filter prepared rows by market, year and month. Dates appear as year and month, followed by the market, calendar month and arrival counts. Download the selected rows as CSV.

**Model:** view the model details, trained-model results and separate simple-rule results.

The app model was trained through September 2026. Earlier evaluation scores do not test this version. Errors vary by market, sudden travel changes can affect results, and no prediction interval is available.

## Checks and MLOps

```sh
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
```

GitHub Actions checks the model and container, publishes the tested image and waits for approval before deploying to Azure. Bicep defines the cloud resources. Logs are enabled; automatic rollback and forecast-error monitoring are not implemented.

For initial setup or a different subscription, sign into Azure and GitHub CLI, then use Bash with `az`, `gh` and `jq`:

```sh
bash infra/setup-github.sh SUBSCRIPTION_ID REGION sbuhary/sri-lanka-tourism-forecast
```

[DagsHub](https://dagshub.com/sbuhary/sri-lanka-tourism-forecast/experiments) holds the original comparisons and app-model run. The added comparisons are in the notebook. To log the original runs again:

```sh
python -m pip install -r requirements-tracking.txt
python src/track_experiments.py
```

The monthly job checks SLTDA on the 1st, 5th and 10th. Changed data creates a review pull request; it does not deploy automatically. Review the data and rerun the notebook when needed. To check the source manually:

```sh
python src/update_data.py --check-only
```

## Files

- `data/`: dataset, country-name mapping and source details.
- `models/`: app model, settings, comparison scores and tracking records.
- `notebooks/`: one analysis notebook, including explanations.
- `src/`: data updates, training, forecasts and tracking.
- `tests/`: data, model and app checks.
- `infra/` and `.github/`: cloud setup and workflows.
- `assets/` and `.streamlit/`: banner, styles and theme.

Banner: supplied [SLTDA coastal photograph](https://www.sltda.gov.lk/images/statistics-cover-new.jpg).
