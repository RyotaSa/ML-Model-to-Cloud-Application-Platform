from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds


def compute_model_metrics_on_slices(model, test, encoder, lb, cat_features):
    """
    
    Compute the performance of metrics on categorical features
    
    """
    metric_dict = {}
    for cat_feat in cat_features:
        col_dict = {}
        for col_value in test[cat_feat].unique():
            X, y, _, _ = process_data(
                                    test[test[cat_feat] == col_value],
                                    categorical_features=cat_features,
                                    label='salary',
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            # metric_dict[cat_feat][col_value] = [precision, recall, fbeta]
            col_dict[col_value] = [precision, recall, fbeta]
        metric_dict[cat_feat] = col_dict

    return metric_dict