from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame, Series

def logistic_regression(
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame
):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    return lr.predict(X_test)


def decision_tree(
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame
):
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    return dtree.predict(X_test)

def knn(
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    k = 5
):
    knn5 = KNeighborsClassifier(n_neighbors=k)
    knn5.fit(X_train, y_train)

    return knn5.predict(X_test)
