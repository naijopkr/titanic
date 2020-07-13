import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_dummies(data: pd.DataFrame):
    dummies = [
        'Pclass',
        'Sex',
        'Embarked'
    ]

    return pd.get_dummies(
        data,
        columns=dummies,
        drop_first=True
    )

df_train = get_dummies(pd.read_csv('data/train.csv'))

df_test = get_dummies(pd.read_csv('data/test.csv'))


# Train model
from sklearn.model_selection import train_test_split as tts

names = ['Sex_male', 'Pclass_2', 'Pclass_3', 'Embarked_S', 'Embarked_Q', 'Survived']
predictors = names[:-1]


X_train = df_train[predictors]
X_test = df_test[predictors]
y_train = df_train['Survived']

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

out = pd.DataFrame()
out['PassengerId'] = df_test['PassengerId']
out['Survived'] = y_pred

out.head()
out.to_csv('data/out.csv', index=False)
