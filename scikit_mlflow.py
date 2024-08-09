import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import sklearn
import mlflow 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Annotated, Tuple



def load_data():
    data_path = "datasets\Titanic.csv"  
    df = pd.read_csv(data_path)
    return df


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

   
def feature_engineering(df: pd.DataFrame):
    """ In this we transform the model using train test split, also standarize the model """
    X = df.drop(["Survived"],axis=1)
    y = df['Survived']
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, shuffle=True, random_state=42)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print(type(X_train),type(X_test),type(y_train),type(y_test))
    return X_train, X_test, y_train, y_test



def train(sk_model, X_train,y_train):
    sk_model.fit(X_train,y_train)
    train_acc = sk_model.score(X_train,y_train)
    mlflow.log_metric("train_acc",train_acc)
    print(f"Train_accuracy: {train_acc}")


def evaluate(sk_model,X_test,y_test):
    eval_acc = sk_model.score(X_test,y_test)
    preds = sk_model.predict(X_test)
    
    mlflow.log_metric("eval_acc",eval_acc)
    print(f"Eval_accuracy: {eval_acc}")

    conf_matrix = confusion_matrix(y_test,preds)
    ax = sns.heatmap(conf_matrix, annot=True,fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.savefig("sklearn_conf_matrix.png")
    mlflow.log_artifact("sklearn_conf_matrix.png")



def main():
    sk_model = SVC()
    mlflow.set_experiment("Scikit_learn_experiment")
    with mlflow.start_run():
        df = load_data()
        df = preprocess_data(df)
        X_train,X_test,y_train,y_test = feature_engineering(df)
        train(sk_model,X_train,y_train)
        evaluate(sk_model,X_test,y_test)
        mlflow.sklearn.log_model(sk_model, "log_reg_model")
        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()


if __name__ == "__main__":
    main()