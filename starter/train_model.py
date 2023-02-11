# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle

from ml.data import process_data, data_clean
from ml.model import train_model, compute_model_metrics, inference, compute_model_metrics_on_slices

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
data = data_clean(data)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

with open('../model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('../model/label_binarizer.pkl', 'wb') as f:
    pickle.dump(lb, f)

# Model Inference
preds = inference(model, X_test)

# Model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Model metric on slices
metric_dict = compute_model_metrics_on_slices(model, test, encoder, lb, cat_features)

with open("slice_output.txt", "w") as f:
        f.write("Metrics on slices\n")
        
        for cat_feat, col_dict in metric_dict.items():
            f.write(f"\nCategory name: {cat_feat}\n")
            f.write("value, precision, recall, fbeta\n")
            for value, metrics in col_dict.items():
                f.write(f"\n{value},{metrics[0]},{metrics[1]},{metrics[2]}\n")

# Print test metrics
print(precision)
print(recall)
print(fbeta)