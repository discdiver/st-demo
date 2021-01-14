import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_lr_iris():
    """train and save an iris ml model as a pickle file
    
    Args:
        None
    
    Returns:
        None
     """

    # load data
    df_iris = px.data.iris()

    # create X and y
    X = df_iris.drop(columns=["species", "species_id"])
    y = df_iris["species"]

    # TTS
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    # instantiate and fit logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # save the fit model for later use
    with open("saved_iris_model2.pkl", "wb") as file:
        pickle.dump(lr, file)


if __name__ == "__main__":
    train_lr_iris()
