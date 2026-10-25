#!/usr/bin/env bash
set -euo pipefail

# Pass the account and image explicitly so the same script works in another subscription.
subscription="${1:?Usage: bash infra/deploy.sh SUBSCRIPTION_ID REGION IMAGE [RESOURCE_GROUP]}"
region="${2:?Provide an Azure region}"
image="${3:?Provide an image tagged with its source commit or digest}"
group="${4:-rg-tourism-forecast}"

az account set --subscription "$subscription"
for provider in Microsoft.App Microsoft.OperationalInsights; do
  az provider register --namespace "$provider" --wait
done
az group create --name "$group" --location "$region" --output none
az deployment group create --resource-group "$group" \
  --name tourism-forecast \
  --template-file "$(dirname "$0")/main.bicep" \
  --parameters location="$region" image="$image" \
  --query properties.outputs --output json
