# Sri Lanka Tourist Arrivals Forecast

[Open the live app](https://sri-lanka-tourism-forecast.streamlit.app/)

Predict monthly arrivals to Sri Lanka from an origin market using its recent arrival counts.

## Run the app

Use Python 3.13. From this folder:

```sh
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

The saved model is included, so training is only needed to rebuild it. The dataset is `Tourism_Arrivals.csv`. The analysis is in [Tourism_Arrivals_Analysis.ipynb](Tourism_Arrivals_Analysis.ipynb). Run its setup cell first; it uses the same CSV in this folder.

## Model and results

I compared models trained on 2019–2023 data and validated on 2024. Random Forest had the lowest MAE: 241.50 arrivals per origin/month, with R² 0.921 and WAPE 25.96%. These are development-validation results, not an independent final test.

I then retrained the selected settings on all 17,053 eligible rows through September 2026: 150 trees, maximum depth 12, minimum leaf size 5 and random seed 42. The saved pipeline includes encoding and scaling. `model_info.json` records the training range, dataset hash and default inputs. This deployment model has not been separately tested.

The six inputs are country, calendar month and arrivals 1, 2, 3 and 12 months earlier. Year identifies the target period but is not a model input.

## Default forecast

The form starts with India, October 2026:

| Historical month | Arrivals |
| --- | ---: |
| September 2026 | 51,931 |
| August 2026 | 47,253 |
| July 2026 | 44,547 |
| October 2025 | 48,113 |

The forecast is about 50,522 arrivals. Change all four historical counts when choosing another origin or target period; the app does not fetch them automatically.

## Data and limitations

The dataset was prepared from [official SLTDA country-arrival tables](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries). Each row contains an origin/month, four historical arrival counts, calendar month and actual arrivals. Clear country-name variations were standardised; unresolved records and rows without the required history were excluded. The source tables cover 2018 to September 2026; 2018 provides the first annual lags.

Errors vary by market. Publication delays, revisions and sudden changes in travel demand can affect forecasts. The app does not provide prediction intervals. Estimates for periods used in training are not validation results.

## Automated checks

```sh
python -m unittest discover -s tests -v
```

GitHub Actions rebuilds the model and runs these checks on pushes to `main` and pull requests. It checks the dataset, saved prediction and app, then stores the model and training record as an artifact linked to the commit. Code, data and the current model are versioned together in Git.

For new monthly data, update the dataset and training cutoff, check the defaults and rerun training and tests before release. The fixed forecast check must be reviewed if the data or settings change. Do not replace it just to hide a failed check.

These checks confirm that the app can be reproduced and run; they do not establish forecast accuracy. The app is hosted on Streamlit Community Cloud using Python 3.13. Live monitoring is not implemented. Actual forecast errors can be checked once SLTDA publishes the corresponding arrivals.


## Experiment tracking

[DagsHub project and experiments](https://dagshub.com/sbuhary/sri-lanka-tourism-forecast/experiments)

```sh
python -m pip install -r requirements-tracking.txt
python track_experiments.py
```

Commit local changes first so each run points to the code that produced it. The DagsHub client prompts for browser sign-in when needed. Credentials stay outside the repo.

The script reproduces the notebook comparisons on 2024 development data and records parameters, metrics, the data hash and source commit. A separate deployment run stores the refitted model and its training record, without assigning the validation scores to that refit. Tracking dependencies are separate from the app requirements.
