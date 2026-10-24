param location string = resourceGroup().location
param repository string
param ownerId string
param repositoryId string
param environmentName string = 'production'

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'tourism-github-deploy'
  location: location
}
resource trust 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: identity
  name: 'github-production'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: 'repo:${split(repository, '/')[0]}@${ownerId}/${split(repository, '/')[1]}@${repositoryId}:environment:${environmentName}'
    audiences: ['api://AzureADTokenExchange']
  }
}
// Deployment can manage this resource group, but cannot grant Azure roles.
resource deployRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, identity.id, 'contributor')
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
  }
}
output clientId string = identity.properties.clientId
output tenantId string = tenant().tenantId
output subscriptionId string = subscription().subscriptionId
