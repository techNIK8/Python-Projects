import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import warnings

# Dataframe = df
df = pd.read_csv('winequality-red.csv')

# print(df.shape)
# print(df.head)
# print(df.info()) # Τι έχει αποθηκευτεί στο df και τα χαρακτηριστικά του.

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# plt.show()

# print(df['quality'].value_counts())


X = df.drop(['quality'], axis=1)
y = df['quality']

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# print(y)

# 75% training and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# default 'rbf'
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

# making confusing matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='2.0f')
# plt.show()


warnings.filterwarnings('ignore')

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

# Model F1 score
print("F1 score:", format(metrics.f1_score(y_test, y_predict, average='weighted'), '.3f'))

# Model Precision
print("Precision:", format(metrics.precision_score(y_test, y_predict, average='weighted'), '.3f'))

# Model Recall
print("Recall:", format(metrics.recall_score(y_test, y_predict, average='weighted'), '.3f'))
