import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_test.head()
print(df.head())

# EDA
sns.heatmap(
    df.isnull(),
    yticklabels=False,
    cbar=False,
    cmap='viridis'
)

sns.set_style('whitegrid')
sns.countplot(
    x='Survived',
    hue='Sex',
    data=df,
    palette='RdBu_r'
)

sns.countplot(
    x='Survived',
    hue='Pclass',
    data=df,
    palette='rainbow'
)

sns.distplot(
    df['Age'].dropna(),
    kde=False,
    color='darkred',
    bins=30
)

sns.countplot(
    x='SibSp',
    data=df
)

sns.distplot(
    df['Fare'],
    color='green',
    kde=False,
    bins=40
)

# Data Cleaning

sns.boxplot(
    x='Pclass',
    y='Age',
    data=df,
    palette='winter'
)

first_class_age = df[df['Pclass'] == 1]['Age'].mean()
second_class_age = df[df['Pclass'] == 2]['Age'].mean()
third_class_age = df[df['Pclass'] == 3]['Age'].mean()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return first_class_age

        elif Pclass == 2:
            return second_class_age

        else:
            return third_class_age

    else:
        return Age

df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
df_test['Age'] = df_test[['Age', 'Pclass']].apply(impute_age, axis=1)

df.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)

df.head()
df.dropna(inplace=True)
df_test['Fare'].head()
df_test.at[152, 'Fare'] =  df_test['Fare'].mean()

df.info()
df_test.info()

sex = pd.get_dummies(df['Sex'], drop_first=True)
embark = pd.get_dummies(df['Embarked'], drop_first=True)
df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
df = pd.concat([df, sex, embark], axis=1)

sex_test = pd.get_dummies(df_test['Sex'], drop_first=True)
embark_test = pd.get_dummies(df_test['Embarked'], drop_first=True)
df_test.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
df_test = pd.concat([df_test, sex_test, embark_test], axis=1)

df.head()
df_test.head()

# Building logistic regression model

X_train = df.drop('Survived', axis=1)
y_train = df['Survived']

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(df_test)

output = pd.DataFrame()
output['PassengerId'] = df_test['PassengerId']
output['Survived'] = predictions

output.head()
output.to_csv(
    'data/2020061515_titanic_survivor.csv',
    encoding='utf-8',
    index=False
)
