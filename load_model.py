import mlflow
import pandas as pd
import numpy as np

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001/"
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = "runs:/172cfe04e5724a1daaf3299f19080634/iris_model"

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
data=np.array([[1.0,2.0,3.0,4.0]])

# TODO: Predict on a Pandas DataFrame. Due to the MLFlow functionality constrain.
#       The loaded model's predict function only accept dataframe as input instead of numpy array.
prediction=loaded_model.predict(pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4']))

# Print out prediction result
print(prediction)
