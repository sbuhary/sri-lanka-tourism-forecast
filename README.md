# Sri Lanka Tourist Arrivals Forecast

Predict monthly arrivals to Sri Lanka from an origin market using Extra Trees.

[Azure app](https://tourism-forecast.calmisland-751fb489.eastus.azurecontainerapps.io/) | [Streamlit app](https://sri-lanka-tourism-forecast.streamlit.app/) | [Notebook](notebooks/Tourism_Arrivals_Analysis.ipynb)

## Run locally

Use Python 3.13 from the repo root:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved model is included. To retrain it, run `python src/train_model.py`. The notebook finds the CSV in `data/` or beside itself. In Colab, upload it when asked.

## Data and model

The dataset has 17,053 records from January 2019 to September 2026, one market per month. It comes from nine [official SLTDA Excel files](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries). The 2018 counts supply the first annual lag. Clear country-name differences were corrected, zeros were kept, and unresolved entries or rows with missing history were removed.

The six inputs are market, calendar month and arrivals 1, 2, 3 and 12 months earlier. Year labels the period but is not an input.

Extra Trees averages 150 trees with random split thresholds. Each tree uses all training rows and encoded features. A check on 2023 selected depth 8 and minimum leaf size 10. The tree count was fixed. See [Geurts et al. (2006)](https://doi.org/10.1007/s10994-006-6226-1) and [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

## Results

Training on 2019–2023 and validating on 2024 gave Extra Trees MAE **258.39**, RMSE **1,111.30**, R² **0.904** and WAPE **27.77%**. Random Forest had lower MAE (**241.50**), followed by voting (**243.63**). Extra Trees is the main algorithm studied here, but it was not the best model in this comparison. R² is not percentage accuracy.

After training through 2024, I checked January 2025–September 2026. MAE was **293.35** for Extra Trees, **299.55** for the previous-month rule and **219.67** for the same-month-last-year rule. I had already examined this period, so it is not an independent final test. These checks use actual earlier counts.

Extra Trees beats the previous-month rule for only 57 of 184 validation markets. The notebook includes the market errors and SHAP/LIME explanations. [Evaluation records](models/evaluation_results.json) hold the settings and scores.

## Using the app

Choose a market and a one-, two- or three-month horizon. The page scrolls to the results after a selection. It stays at the top on initial load.

The first forecast uses actual historical counts. Later months reuse earlier predictions where actual counts are unavailable. The chart shows 12 published months followed by forecasts. On narrow screens, the chart opens at the latest months; swipe right to see earlier history. Hover for exact counts, or download the forecast CSV.

Open **View or adjust starting inputs** to edit a count. Save applies it to this scenario. Reset restores the dataset value; changing market clears edits. Missing required history stops the forecast. The chart history and percentage comparisons always use published actuals.

The default India forecasts for October–December 2026 are **44,499 / 44,079 / 43,290**. The app model was trained through September 2026. Earlier scores do not test this trained version. Current dates and settings are in `models/model_info.json`.

A separate three-month check started forecasts in January, April, July and October 2024 with the model trained through 2023. MAE rose from **286.69** at one month ahead to **353.78** and **386.80** at two and three months (710, 709 and 709 cases). The seasonal rule gave **349.45 / 322.34 / 266.50**, so it was better at the longer horizons. Future actuals were excluded from inputs. See [horizon results](models/horizon_results.json).

These are planning estimates. Errors vary by market, and no prediction interval is provided. The data is public monthly totals, not personal records. Revisions and sudden demand changes can affect results, so forecasts need human review.

## Checks and tracking

```sh
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
python -m pip install -r requirements-tracking.txt
python src/track_experiments.py
```

Commit changes before tracking. [DagsHub](https://dagshub.com/sbuhary/sri-lanka-tourism-forecast/experiments) holds seven validation runs, three later-period runs and a separate app-model run. Run IDs and code versions are in [tracking_results.json](models/tracking_results.json). Tracking is manual. Credentials stay outside the repo.

## Azure deployment

GitHub Actions runs model and container checks, publishes the tested image to GHCR and waits for approval. It then signs into Azure with OIDC, deploys using Bicep and checks the live app. See the [verified release](https://github.com/sbuhary/sri-lanka-tourism-forecast/actions/runs/40202957949).

For initial setup or a new subscription, sign into Azure and GitHub CLI. Use Bash with `az`, `gh` and `jq`:

```sh
bash infra/setup-github.sh SUBSCRIPTION_ID REGION sbuhary/sri-lanka-tourism-forecast
```

This creates the resource group, Azure identity and GitHub approval settings. The account needs permission to assign Azure roles. The deployment identity has Contributor access to this resource group. The signed-in GitHub user is the reviewer. Self-review is allowed for this single-maintainer project; only `main` can deploy.

For a manual release or rollback to a tested image:

```sh
bash infra/deploy.sh SUBSCRIPTION_ID REGION ghcr.io/sbuhary/sri-lanka-tourism-forecast:COMMIT_SHA
```

The accounts, subscription and public GHCR image must exist. Bicep sets up HTTPS, a readiness check, zero–one replica and Log Analytics. Logs have 30-day retention and a 0.1 GB daily cap. Usage can consume cloud credit. To move subscriptions, rerun setup, verify the new app, then remove old resources when appropriate.

Logs are enabled. Automatic forecast-error monitoring, drift-triggered training and rollback are not implemented.

## Monthly data updates

GitHub Actions checks SLTDA on the 1st, 5th and 10th at 08:47 Sri Lanka time. You can also use **Actions > Update SLTDA data > Run workflow**. GitHub may delay scheduled runs or disable schedules after 60 days without repository activity.

The job downloads Excel files, checks totals and country labels, then rebuilds the historical features. Blank counts stay missing. Source-format or total mismatches stop the update.

Changed data produces a candidate model with the existing settings and a review pull request. Source URLs and hashes are saved in the manifest. Downloads and validation details are kept as workflow artifacts for 90 days.

Before merging, review changes and exclusions. Rerun the notebook if historical values changed, and log approved model updates to DagsHub. Merging starts the checks; Azure still needs approval. The job does not tune models or replace historical evaluation scores.

To check the source without changing the CSV:

```sh
python -m pip install -r requirements-data.txt
python src/update_data.py --check-only
```

The setup script enables GitHub Actions to create pull requests. No SLTDA account is needed.

## Files

```text
app.py       Streamlit app
assets/      Banner and styles
.streamlit/  Theme
src/         Data updates, training, forecasts and tracking
data/        Dataset and source details
models/      Saved model, settings and results
notebooks/   Analysis and explanations
tests/       Automated checks
infra/       Azure configuration
.github/     Check, update and deployment workflows
```

Banner: supplied SLTDA coastal photograph, [source](https://www.sltda.gov.lk/images/statistics-cover-new.jpg).

## Additional comparisons

`notebooks/Tourism_Model_Comparison.ipynb` compares 13 ensemble approaches and two baselines. It has its own installation cell; the app does not need these extra libraries. The main notebook includes a short results summary. Extra Trees remains the deployed model; the comparison vote has lower 2024 MAE, while the seasonal baseline is stronger in the later period and at the third forecast step.
