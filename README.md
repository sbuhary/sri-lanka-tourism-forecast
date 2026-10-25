# Sri Lanka Tourist Arrivals Forecast

Estimate monthly tourist arrivals to Sri Lanka from an origin market using recent arrival counts.

[Azure app](https://tourism-forecast.calmisland-751fb489.eastus.azurecontainerapps.io/) | [Streamlit app](https://sri-lanka-tourism-forecast.streamlit.app/) | [Notebook](notebooks/Tourism_Arrivals_Analysis.ipynb)

## Run locally

Use Python 3.13 from the repo root:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved model is included. To rebuild it, run `python src/train_model.py`. The notebook loads the CSV from `data/` or lets you upload it in Colab.

## Data and model

The dataset contains 17,053 origin-month records from January 2019 to September 2026. It was compiled from nine [official SLTDA tables](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries). The 2018 data supplies the first annual lag. Clear country-name variations were standardised; genuine zeros were retained, and unresolved records or rows without enough history were excluded.

The six inputs are origin market, calendar month and arrivals 1, 2, 3 and 12 months earlier. Year labels the period but is not a model input.

Extra Trees averages 150 trees with random split thresholds. Every tree uses the full training sample and all encoded features. A 2023 holdout selected depth 8 and minimum leaf size 10. See [Geurts et al. (2006)](https://doi.org/10.1007/s10994-006-6226-1) and [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

## Results

After training on 2019–2023, Extra Trees had MAE **258.39**, RMSE **1,111.30**, R² **0.904** and WAPE **27.77%** on 2024 validation. Random Forest had lower MAE at **241.50**, followed by voting at **243.63**. Extra Trees is the main algorithm investigated here, while the comparison shows where other methods work better. R² is not percentage accuracy.

For a later check, the fixed settings were refitted through 2024 and assessed on January 2025–September 2026. MAE was **293.35**, compared with **299.55** for the previous-month rule and **219.67** for the same-month-last-year rule. This later period was used during exploration, so it is not an independent final test. Each prediction uses actual earlier counts.

Extra Trees improves on the previous-month rule for 57 of 184 origins in validation. Errors vary by market, and the stronger seasonal baseline in the later period limits the case for operational use. The notebook contains the full comparison and SHAP/LIME explanations. [Evaluation records](models/evaluation_results.json) retain the settings and results.

## App example

The app model was retrained on all eligible records through September 2026. Earlier scores describe the evaluation models, not this deployment refit. The default India October 2026 forecast is **44,499**, using these published counts:

| Historical month | Arrivals |
| --- | ---: |
| September 2026 | 51,931 |
| August 2026 | 47,253 |
| July 2026 | 44,547 |
| October 2025 | 48,113 |

Update all four counts when changing the origin or month. The app does not fetch them automatically. Publication delays, revisions and sudden demand changes can affect estimates. No prediction interval is provided.

The data is public and aggregated, with no individual travel records. Country mappings and uneven market errors can affect budget or service decisions, so forecasts need human review.

## Checks and experiment tracking

```sh
python -m unittest discover -s tests -v
python -m pip install -r requirements-tracking.txt
python src/track_experiments.py
```

Commit changes before tracking so each run identifies its source code. [DagsHub](https://dagshub.com/sbuhary/sri-lanka-tourism-forecast/experiments) stores seven validation runs, three later-period runs and a separate deployment artifact. Run IDs and source revisions are in [tracking_results.json](models/tracking_results.json). Tracking runs manually; credentials stay outside the repo.

## Azure deployment

GitHub Actions requires the full model checks and container checks to pass. It publishes the tested image to GHCR, waits for production approval, signs into Azure through OIDC, deploys the Bicep configuration and checks live health. The [Extra Trees release](https://github.com/sbuhary/sri-lanka-tourism-forecast/actions/runs/40174959817) completed successfully.

For initial setup or migration, sign into Azure and GitHub CLI, then use Bash with `az`, `gh` and `jq`:

```sh
bash infra/setup-github.sh SUBSCRIPTION_ID REGION sbuhary/sri-lanka-tourism-forecast
```

The setup creates the resource group, deployment identity, repository/environment trust and GitHub approval settings. It needs an account allowed to assign Azure roles. The deployment identity has Contributor on this resource group. The signed-in GitHub user becomes reviewer; self-review is allowed for this single-maintainer project. Only `main` can deploy.

For a manual deployment or rollback to a tested image:

```sh
bash infra/deploy.sh SUBSCRIPTION_ID REGION ghcr.io/sbuhary/sri-lanka-tourism-forecast:COMMIT_SHA
```

The subscription, accounts and public GHCR image must already exist. Bicep configures Container Apps, HTTPS, a readiness probe, zero–one replica and Log Analytics. Logs have 30-day retention and a 0.1 GB daily ingestion cap. Usage can consume cloud credit. To change subscription, rerun setup with the new ID, verify the new app and then remove old resources when appropriate.

Container logs are enabled. Forecast-error monitoring and automatic retraining are not implemented. A failed deployment health check does not automatically roll back the app.

## Monthly data updates

GitHub Actions checks SLTDA on the 1st, 5th and 10th of each month at 08:47 Sri Lanka time. It can also be started from **Actions > Update SLTDA data > Run workflow**. GitHub can delay scheduled runs and disables schedules in public repositories after 60 days without activity.

The job downloads the country-level Excel files, checks published totals, applies the same country mappings and exclusions, then rebuilds the historical features. It detects revised values as well as new months. Blank counts stay missing. Source layout or reconciliation errors stop the job without publishing a data update.

When data changes, the job refits a candidate model with the existing settings and opens or updates a pull request. The latest available month determines the training cutoff and next-month example. The source manifest records URLs and SHA-256 hashes. Original downloads and the validation report are retained as workflow artifacts for 90 days; download them for longer retention.

Review the change summary and exclusions, rerun the notebook if historical values changed, and log an approved model update to DagsHub using the tracking script. Merging starts the existing container and model checks. Azure deployment still waits for production approval. This job does not retune models or update historical evaluation scores.

To check the source locally without changing the CSV:

```sh
python -m pip install -r requirements-data.txt
python src/update_data.py --check-only
```

GitHub Actions must be allowed to create pull requests in the repository's Actions settings. No SLTDA account or additional cloud service is needed.

## Files

```text
app.py       Streamlit frontend
src/         Training and tracking scripts
data/        Dataset
models/      Model, metadata and result records
notebooks/   Analysis and explanations
tests/       Automated checks
infra/       Azure configuration
.github/     Check and deployment workflows
```
