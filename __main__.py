import pandas as pd
import seaborn as sns

df = pd.read_csv('data/clean_train.csv')

from sklearn.model_selection import train_test_split as tts

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = tts(
    X,
    y,
    test_size=0.3,
    random_state=101
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train, y_train)

y_pred = kmeans.predict(X_test)
accuracy_score(y_test, y_pred)
