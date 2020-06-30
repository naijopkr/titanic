import pandas as pd
from pandas import DataFrame


def impute_age(cols, pclass=(0,0,0)):
    Age = cols[0]
    Pclass = cols[1]

    class1, class2, class3 = pclass

    if pd.isnull(Age):

        if Pclass == 1:
            return class1

        elif Pclass == 2:
            return class2

        else:
            return class3

    else:
        return Age


def get_age_avg(df: DataFrame):
    class1 = df[df['Pclass'] == 1]['Age'].mean()
    class2 = df[df['Pclass'] == 2]['Age'].mean()
    class3 = df[df['Pclass'] == 3]['Age'].mean()

    pclass = (class1, class2, class3)

    df['Age'] = df[['Age', 'Pclass']].apply(
        lambda cols: impute_age(cols, pclass
    ), axis=1)


def prepare_data(df: DataFrame):
    get_age_avg(df)

    sex = pd.get_dummies(df['Sex'], drop_first=True)
    embark = pd.get_dummies(df['Embarked'], drop_first=True)
    clean_data = df.drop(['Sex', 'Embarked', 'Name', 'Ticket', "Cabin", "PassengerId"], axis=1)

    return  pd.concat([clean_data, sex, embark], axis=1)
