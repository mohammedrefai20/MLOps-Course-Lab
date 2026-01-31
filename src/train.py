"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    mlflow.sklearn.log_model(
            sk_model=col_transf,
            artifact_path="preprocessor"
        )
    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    # log_reg = LogisticRegression(max_iter=1000)
    # svm = SVC(C=1.0,kernel='rbf',gamma=10)
    dt=DecisionTreeClassifier(criterion='gini',max_depth=5)
    # svm.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    # log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, dt.predict(X_train))
    # Log model
    # with mlflow.start_run(run_name="logreg_training"):
    mlflow.sklearn.log_model(
        sk_model=dt,
        name="model",
        signature=signature,
        input_example=X_train.head(2)
    )
    ### Log the data
    X_train.to_csv("X_train.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    mlflow.log_artifact("X_train.csv")
    mlflow.log_artifact("y_train.csv")
    return dt


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    ### Set the experiment name
    mlflow.set_experiment("Churn_Experiment")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="Churn_DT_Run"):
        df = pd.read_csv("G:\ITI_AI\Lectures\MLflow\Assignments\Project\MLOps-Course-Lab\dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter
    #     svm = svm.SVC(C=1.0,kernel='rbf',gamma=10)
    # dt=DecisionTreeClassifier(criterion='gini',max_depth=5)

        # mlflow.log_param("max_iter",1000)
        mlflow.log_params({"criterion":"gini",
                           "max_depth":5,
                           })
        

        model = train(X_train, y_train)
        y_pred = model.predict(X_test)

    ### Log metrics after calculating them
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

    ### Log tag
    mlflow.set_tag("model_type", "Decision Tree")



    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    
    # Log the image as an artifact in MLflow
    fig_path = "confusion_matrix.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    plt.show()


if __name__ == "__main__":
    main()
