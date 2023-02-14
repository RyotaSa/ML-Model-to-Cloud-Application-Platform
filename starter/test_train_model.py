import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from ml.data import process_data, data_clean
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
data = data_clean(data)

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

model = train_model(X_train, y_train)
preds = inference(model, X_test)

# Test checking encoder and label binarizer
def test_encoder_and_lb(encoder, lb):
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer

# Test if prediction values exist
def test_prediction_exist(preds):
    assert len(preds) > 0

# Test metrics if values are between 0 and 1
def test_metrics(y_test, preds):
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 < precision < 1
    assert 0 < recall < 1
    assert 0 < fbeta < 1

test_encoder_and_lb(encoder, lb)
test_prediction_exist(preds)
test_metrics(y_test, preds)