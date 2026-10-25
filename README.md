# Sri Lanka Tourist Arrivals Forecast

[Open the Azure app](https://tourism-forecast.calmisland-751fb489.eastus.azurecontainerapps.io/) · [Streamlit fallback](https://sri-lanka-tourism-forecast.streamlit.app/)

Predict monthly arrivals to Sri Lanka from an origin market using its recent arrival counts.

## Run the app

Use Python 3.13. From this folder:

```sh
python -m pip install -r requirements.txt
python src/train_model.py
python -m streamlit run app.py
```

The saved model is included, so training is only needed to rebuild it. The dataset is `data/Tourism_Arrivals.csv`. The analysis is in [Tourism_Arrivals_Analysis.ipynb](notebooks/Tourism_Arrivals_Analysis.ipynb). Run its setup cell first; it loads the CSV from `data/`, or lets you upload it in Colab.

## Model and results

Extra Trees is the algorithm under study. It averages 150 trees with randomly sampled split thresholds. Here `bootstrap=False` and `max_features=1.0`: each tree uses the full training sample and all available features. A limited 2023 holdout check selected depth 8 and minimum leaf size 10. See the [original paper](https://doi.org/10.1007/s10994-006-6226-1) and [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

On 2024 development validation after training on 2019–2023, Extra Trees has MAE **258.39**, RMSE **1,111.30**, R² **0.904** and WAPE **27.77%**. Random Forest is better at MAE **241.50**, and voting is **243.63**. Extra Trees improves on the previous-month rule (286.45) for the pooled metric, but only for **57 of 184 origins**. It is not the winner of this comparison. R² is not percentage accuracy.

For a separate retrospective test, the fixed settings were refitted on 13,066 records through 2024 and assessed on 3,978 known-origin records from January 2025 to September 2026. Extra Trees MAE is **293.35**, previous-month MAE **299.55**, and same-month-last-year MAE **219.67**. The seasonal rule is better there. Those later data were previously inspected, so this is not an untouched final test, and it was not used to change the settings. Every forecast uses actual earlier counts, so evaluation is rolling one month ahead rather than a fixed twelve-month forecast.

[Evaluation results](models/evaluation_results.json) retain the settings, cohort sizes and metrics. The deployment pipeline refits Extra Trees on all 17,053 eligible records through September 2026. Earlier validation and retrospective scores do not evaluate this refit. `models/model_info.json` records its data hash, settings and default inputs.

## Default forecast

The form starts with India, October 2026:

| Historical month | Arrivals |
| --- | ---: |
| September 2026 | 51,931 |
| August 2026 | 47,253 |
| July 2026 | 44,547 |
| October 2025 | 48,113 |

The forecast is about 44,499 arrivals. Change all four historical counts when choosing another origin or target period; the app does not fetch them automatically.

## Data and limitations

The dataset was prepared from [official SLTDA country-arrival tables](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries). Each row contains an origin/month, four historical arrival counts, calendar month and actual arrivals. Clear country-name variations were standardised; unresolved records and rows without the required history were excluded. The source tables cover 2018 to September 2026; 2018 provides the first annual lags.

Errors vary by market. Publication delays, revisions and sudden changes in travel demand can affect forecasts. The app does not provide prediction intervals. Estimates for periods used in training are not validation results.

## Automated checks

```sh
python -m unittest discover -s tests -v
```

GitHub Actions rebuilds the model and runs these checks on relevant pushes to `main`, pull requests and manual runs. The release workflow calls the same checks as a required job. It checks the dataset, saved prediction and app, then stores the model and training record as an artifact linked to the commit. Code, data and the current model are versioned together in Git.

For new monthly data, update the dataset and training cutoff, check the defaults and rerun training and tests before release. The fixed forecast check must be reviewed if the data or settings change. Do not replace it just to hide a failed check.

These checks confirm that the app can be reproduced and run; they do not establish forecast accuracy. Azure serves the last approved release; approve the latest successful build to update it. The app runs on Azure Container Apps, with Streamlit Community Cloud as a fallback. Both use Python 3.13. Azure collects container logs; forecast-error monitoring is not implemented. Actual forecast errors can be checked once SLTDA publishes the corresponding arrivals.


## Experiment tracking

[DagsHub project and experiments](https://dagshub.com/sbuhary/sri-lanka-tourism-forecast/experiments)

```sh
python -m pip install -r requirements-tracking.txt
python src/track_experiments.py
```

Commit local changes first so each run points to the code that produced it. The DagsHub client prompts for browser sign-in when needed. Credentials stay outside the repo.

The script reproduces the notebook comparisons on 2024 development data and records parameters, metrics, the data hash and source commit. A separate deployment run stores the refitted model and its training record, without assigning the validation scores to that refit. Tracking dependencies are separate from the app requirements.

Historical Random Forest runs remain in the original DagsHub experiment. The new Extra Trees experiment is configured to record seven validation comparisons, three retrospective assessments and a separate deployment artifact. Until new tracking completes, tracking_results.json is explicitly labelled historical. [tracking_results.json](models/tracking_results.json) records their IDs, metrics and source revision. Do not assign old Random Forest metrics or artifact checks to the updated Extra Trees model. Tracking is run on demand; GitHub Actions currently rebuilds and tests the app without logging new DagsHub runs.


## Azure deployment

The original Random Forest deployment was verified in East US on 24 October 2026. The Extra Trees release passed [both model and container checks](https://github.com/sbuhary/sri-lanka-tourism-forecast/actions/runs/40174959817) on 25 October and is awaiting production approval. Its verified local default forecast is 44,499; do not treat the previous Azure release as the updated model until deployment succeeds.

The initial deployment used a personal Azure trial; the same template can recreate it in another subscription.

The container workflow builds and checks the app before publishing an image to GitHub Container Registry, tagged with its source commit. Make the image package public for Azure to pull it without a registry credential. The image contains only the app, dependencies and verified model files.

In Azure Cloud Shell (Bash), clone this repository and run:

```sh
bash infra/deploy.sh SUBSCRIPTION_ID REGION ghcr.io/sbuhary/sri-lanka-tourism-forecast:COMMIT_SHA
```

`infra/main.bicep` creates a Container Apps Consumption environment, the app and a Log Analytics workspace. Scaling is limited to zero–one replica; logs have 30-day retention and a 0.1 GB daily ingestion cap. These settings reduce usage but do not guarantee zero cost. No Azure Container Registry or virtual machine is required.

To use another subscription later, sign into it and run the same script with its subscription ID and an available region. Reuse the tested image. Deployment permissions and the app URL may change. Verify the new app before removing the previous resources; historical logs remain in the original workspace.

To update or roll back, rerun the script with the required tested image tag. Subscription IDs and credentials are not stored in this repo. GitHub Actions runs the full model checks and container checks as separate required jobs. Both must succeed for the same commit before deployment can wait for production approval. After approval it signs into Azure through OIDC, applies the Bicep template with the image digest and checks the live health endpoint. Only `main` can deploy. The reviewer is the account that ran setup; self-review remains allowed for this single-maintainer project. Concurrent releases are queued.

For initial setup or migration, sign into Azure and GitHub CLI with an account allowed to create identities and assign roles, then run in Bash (with `az`, `gh` and `jq` installed):

```sh
bash infra/setup-github.sh SUBSCRIPTION_ID REGION sbuhary/sri-lanka-tourism-forecast
```

This creates a user-assigned identity, trusts only this repository's `production` environment using its immutable owner and repository IDs, grants Contributor on this resource group and configures GitHub's approval, branch policy and environment variables. No Azure password or client secret is stored in GitHub. The deployment identity cannot grant roles. The subscription, account eligibility, GitHub repository and public GHCR package are prerequisites; this script does not create them.

The first deploy after setup can be started from Actions → Build and deploy forecast → Run workflow. Future changes to app/container/infrastructure files on `main` start it automatically. Approve the pending production deployment in Actions after reviewing the build. Tracking in DagsHub remains on demand. Releases use single revisions; canary rollout, automatic rollback, drift monitoring and retraining are not implemented.


## Project layout

```text
app.py          Streamlit frontend
src/            Training and experiment tracking
data/           Consolidated dataset
models/         Saved model, metadata and tracking references
notebooks/      Analysis and model comparison
tests/          Data, inference and app checks
infra/          Azure resources and setup scripts
.github/        CI and approved deployment workflows
```


## Explanation and ethical use

The evaluation-model India January 2024 prediction is 28,951 versus 34,399 observed. Previous-month arrivals has the largest SHAP contribution, about +18,343 relative to the background. LIME has local R² 0.936 and predicts about 1,735 arrivals below the ensemble. That local fit is not overall forecast accuracy. Neither explainer establishes causality, and separately perturbed lags can be unrealistic.

These are public aggregate counts rather than personal travel records. Publication terms and source acknowledgement still matter. Data revisions and country mappings affect quality. Uneven market errors could distort budget or service allocation, so retain human review and market-level error checks. The later seasonal-baseline result is a reason not to replace simple benchmarks automatically.
