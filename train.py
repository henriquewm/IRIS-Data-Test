# %%
# Import necessary libraries
import pandas as pd
import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the data into training and testing sets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(clf, 'iris_model.pkl')
# %%
# Enter details of your Azure Machine Learning workspace
subscription_id = "0f33db52-3637-4aba-8a83-fc6c80c5b8b9"
resource_group = "Henrique_Personal"
workspace = "Henrique_Personal"

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# %%
# Define an endpoint name
import datetime
endpoint_name = "endpt-" + datetime.datetime.now().strftime("%m%d%H%M%f")
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="this is a sample endpoint",
    auth_mode="key"
)

# %%
model = Model(path='iris_model.pkl')
env = Environment(
    conda_file="environment.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

# %%
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code=".", scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
# %%
#Register the model
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

file_model = Model(
    path="iris_model.pkl",
    type=AssetTypes.CUSTOM_MODEL,
    name="iris_model",
    description="Model created from local file.",
)
ml_client.models.create_or_update(file_model)

# %%
#Register the environment
from azure.ai.ml.entities import Environment

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="environment.yaml",
    name="my-env",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)
# %%
#Create the endpoint
ml_client.online_endpoints.begin_create_or_update(endpoint)

# %%
#Create the deployment
blue_deployment_with_registered_assets = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env_docker_conda,
    code_configuration=CodeConfiguration(
        code=".", scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)
# %%
ml_client.online_deployments.begin_create_or_update(blue_deployment_with_registered_assets)
# %%
