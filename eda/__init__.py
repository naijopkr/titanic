import seaborn as sns
import pandas as pd

from pandas import DataFrame

sns.set_style('whitegrid')

# EDA
def check_null(df: DataFrame):
    sns.heatmap(
        df.isnull(),
        yticklabels=False,
        cbar=False,
        cmap='viridis'
    )

def check_na(df: DataFrame):
    sns.heatmap(
        df.isna(),
        yticklabels=False,
        cbar=False,
        cmap='viridis'
    )

def countplot(
    df: DataFrame,
    x='Survived',
    hue=None,
    palette=None
):
    sns.countplot(
        x=x,
        data=df,
        hue=hue,
        palette=palette
    )


def feat_dist(df: DataFrame, feat='Age'):
    sns.distplot(
        df[feat].dropna(),
        kde=False,
        color='darkred',
        bins=30
    )


def feat_count(df: DataFrame, feat='SibSp'):
    sns.countplot(
        x='SibSp',
        data=df
    )


def feat_boxplot(df: DataFrame, x='Pclass', y='Age'):
    sns.boxplot(
        x=x,
        y=y,
        data=df,
        palette='winter'
    )


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
    df.drop('Cabin', axis=1, inplace=True)

    sex = pd.get_dummies(df['Sex'], drop_first=True)
    embark = pd.get_dummies(df['Embarked'], drop_first=True)
    df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
    df = pd.concat([df, sex, embark], axis=1)
