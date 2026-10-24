#!/usr/bin/env bash
set -euo pipefail
subscription="${1:?Usage: bash infra/setup-github.sh SUBSCRIPTION_ID REGION REPOSITORY [RESOURCE_GROUP]}"
region="${2:?Provide a region}"
repository="${3:?Provide owner/repository}"
group="${4:-rg-tourism-forecast}"
az account set --subscription "$subscription"
for provider in Microsoft.App Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
  az provider register --namespace "$provider" --wait
done
az group create --name "$group" --location "$region" --output none
owner_id=$(gh api "repos/$repository" --jq .owner.id)
repo_id=$(gh api "repos/$repository" --jq .id)
identity=$(az deployment group create --resource-group "$group" --name github-identity \
  --template-file "$(dirname "$0")/github-identity.bicep" \
  --parameters repository="$repository" ownerId="$owner_id" repositoryId="$repo_id" --query properties.outputs --output json)
reviewer=$(gh api user --jq .id)
printf '{"reviewers":[{"type":"User","id":%s}],"prevent_self_review":false,"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}' "$reviewer" \
  | gh api --method PUT "repos/$repository/environments/production" --input - >/dev/null
if ! gh api "repos/$repository/environments/production/deployment-branch-policies" --jq '.branch_policies[].name' | grep -qx main; then
  printf '{"name":"main","type":"branch"}' | gh api --method POST \
    "repos/$repository/environments/production/deployment-branch-policies" --input - >/dev/null
fi
gh variable set AZURE_CLIENT_ID --repo "$repository" --env production --body "$(echo "$identity" | jq -r .clientId.value)"
gh variable set AZURE_TENANT_ID --repo "$repository" --env production --body "$(echo "$identity" | jq -r .tenantId.value)"
gh variable set AZURE_SUBSCRIPTION_ID --repo "$repository" --env production --body "$subscription"
gh variable set AZURE_RESOURCE_GROUP --repo "$repository" --env production --body "$group"
gh variable set AZURE_REGION --repo "$repository" --env production --body "$region"
echo 'Setup complete. Push to main, then review the production deployment in Actions.'
