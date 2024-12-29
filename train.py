# %%
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.core import Environment

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
#Register the model in Azure ML
ws = Workspace.from_config(path = "./config.json")
model = Model.register(workspace=ws, model_path="iris_model.pkl", model_name="iris_model")
# %%
#Deploy model
inference_config = InferenceConfig(entry_script='score.py', environment=Environment.from_conda_specification(name='iris-env', file_path='environment.yml'))
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws, name='iris-service', models=[model], inference_config=inference_config, deployment_config=aci_config)
service.wait_for_deployment(show_output=True)


# %%
#Delete service
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# Connect to your Azure ML workspace
ws = Workspace.from_config()

# Get the existing service by name
service_name = 'iris-service'
service = Webservice(name=service_name, workspace=ws)

# Delete the service
service.delete()
print(f"Service {service_name} deleted successfully.")
# %%
