import numpy as np
import numpy
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Annotated, Tuple

from zenml import step, pipeline

@step
def load_data() -> pd.DataFrame:
    data_path = "datasets/Titanic.csv"  
    df = pd.read_csv(data_path)
    return df

@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess data, droping columns that doesnt give much infomation and encoding categorical features"""
    columns_drop = ['PassengerId','Name','Ticket']
    df = df.drop(columns=columns_drop)
    # Convert 'Sex' column to numerical values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Convert 'Embarked' column to dummy variables
    df = pd.get_dummies(df, columns=['Embarked'])
    for column in df.select_dtypes(include=['bool']).columns:
        df[column] = df[column].astype(int)
    print(df.head())

    return df

@step
def feature_engineering(df: pd.DataFrame) -> Tuple[
    Annotated[numpy.ndarray, "X_train"],
    Annotated[numpy.ndarray, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """ In this we transform the model using train test split, also standarize the model """
    X = df.drop(["Survived"],axis=1)
    y = df['Survived']
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, shuffle=True, random_state=42)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


@step
def model() -> ClassifierMixin:
    sk_model = SVC()
    return sk_model

@step
def train(sk_model: ClassifierMixin, X_train: numpy.ndarray,y_train: pd.Series) -> ClassifierMixin:
    sk_model.fit(X_train,y_train)
    train_acc = sk_model.score(X_train,y_train)
    print(f"Train_accuracy: {train_acc}")
    return sk_model


@step
def evaluate(sk_model: ClassifierMixin,X_test: numpy.ndarray,y_test: pd.Series):
    eval_acc = sk_model.score(X_test,y_test)
    preds = sk_model.predict(X_test)

    print(f"Eval_accuracy: {eval_acc}")

    conf_matrix = confusion_matrix(y_test,preds)
    ax = sns.heatmap(conf_matrix, annot=True,fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.savefig("sklearn_conf_matrix.png")


@pipeline
def main():
    df = load_data()
    df = preprocess_data(df)
    X_train,X_test,y_train,y_test = feature_engineering(df)
    sk_model = model()
    sk_model = train(sk_model,X_train,y_train)
    evaluate(sk_model,X_test,y_test)





if __name__ == "__main__":
    main()