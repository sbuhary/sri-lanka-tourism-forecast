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

I compared models trained on 2019–2023 data and validated on 2024. Random Forest had the lowest MAE: 241.50 arrivals per origin/month, with R² 0.921 and WAPE 25.96%. These are development-validation results, not an independent final test.

I then retrained the selected settings on all 17,053 eligible rows through September 2026: 150 trees, maximum depth 12, minimum leaf size 5 and random seed 42. The saved pipeline includes encoding and scaling. `models/model_info.json` records the training range, dataset hash and default inputs. This deployment model has not been separately tested.

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

GitHub Actions rebuilds the model and runs these checks on relevant pushes to `main`, pull requests and manual runs. The release workflow calls the same checks as a required job. It checks the dataset, saved prediction and app, then stores the model and training record as an artifact linked to the commit. Code, data and the current model are versioned together in Git.

For new monthly data, update the dataset and training cutoff, check the defaults and rerun training and tests before release. The fixed forecast check must be reviewed if the data or settings change. Do not replace it just to hide a failed check.

These checks confirm that the app can be reproduced and run; they do not establish forecast accuracy. The app runs on Azure Container Apps, with Streamlit Community Cloud as a fallback. Both use Python 3.13. Azure collects container logs; forecast-error monitoring is not implemented. Actual forecast errors can be checked once SLTDA publishes the corresponding arrivals.


## Experiment tracking

[DagsHub project and experiments](https://dagshub.com/sbuhary/sri-lanka-tourism-forecast/experiments)

```sh
python -m pip install -r requirements-tracking.txt
python src/track_experiments.py
```

Commit local changes first so each run points to the code that produced it. The DagsHub client prompts for browser sign-in when needed. Credentials stay outside the repo.

The script reproduces the notebook comparisons on 2024 development data and records parameters, metrics, the data hash and source commit. A separate deployment run stores the refitted model and its training record, without assigning the validation scores to that refit. Tracking dependencies are separate from the app requirements.

The initial tracked comparison contains seven validation runs (five models and two baselines) and one deployment run. [tracking_results.json](models/tracking_results.json) records their IDs, metrics and source revision. All runs finished successfully; the stored deployment artifact was downloaded and verified against the app model. Tracking is run on demand; GitHub Actions currently rebuilds and tests the app without logging new DagsHub runs.


## Azure deployment

Deployed and checked on 24 October 2026 in East US. The live form returned 50,522 arrivals for the default example, matching the saved model. The deployed image passed the [container checks](https://github.com/sbuhary/sri-lanka-tourism-forecast/actions/runs/40126234359).

Current image digest: `sha256:a96f5e75ae30ffee81408f1e130dbbf8ebd40a53851bd4b875e0efb030b04158`. The first [approved automated release](https://github.com/sbuhary/sri-lanka-tourism-forecast/actions/runs/40129111813) passed OIDC login, Bicep deployment and live health checks. An independent request also returned HTTP 200 (`ok`). The initial deployment uses a personal Azure trial; the same template can recreate it in another subscription.

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
