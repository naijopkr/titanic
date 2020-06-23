import pandas as pd

import eda
import models
import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')
# df_test = pd.read_csv('data/test.csv')

# Data Cleaning
eda.prepare_data(df)


# df_test['Fare'].head()
# df_test.at[152, 'Fare'] =  df_test['Fare'].mean()


X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=101
)


y_lr = models.logistic_regression(X_train, y_train, X_test) # accuracy: 0.7
print('LR')
metrics.print_cm(y_test, y_lr)
metrics.print_cr(y_test, y_lr)
print()

y_dtree = models.decision_tree(X_train, y_train, X_test) # accuracy: 0.63
print('DTree')
metrics.print_cm(y_test, y_dtree)
metrics.print_cr(y_test, y_dtree)

y_knn = models.knn(X_train, y_train, X_test) # accuracy: 0.59
print('KNN')
metrics.print_cm(y_test, y_knn)
metrics.print_cr(y_test, y_knn)
print()










# output = pd.DataFrame()
# output['PassengerId'] = df_test['PassengerId']
# output['Survived'] = predictions

# output.head()
# output.to_csv(
#     'data/2020061515_titanic_survivor.csv',
#     encoding='utf-8',
#     index=False
# )
