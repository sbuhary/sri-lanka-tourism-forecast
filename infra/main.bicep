param location string = resourceGroup().location
param appName string = 'tourism-forecast'
param image string

resource logs 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: json('0.1') }
  }
}

resource environment 'Microsoft.App/managedEnvironments@2025-07-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2025-07-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [{
        name: 'forecast'
        image: image
        resources: {
          cpu: json('0.5')
          memory: '1Gi'
        }
        probes: [{
          type: 'Readiness'
          httpGet: {
            path: '/_stcore/health'
            port: 8501
          }
          initialDelaySeconds: 10
          periodSeconds: 10
        }]
      }]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http'
            http: {
              metadata: { concurrentRequests: '10' }
            }
          }
        ]
      }
    }
  }
}

output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output logWorkspace string = logs.name
