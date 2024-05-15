from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import warnings


def SVM_calc_fun(dfBX):
    print("SVM calculation is running")

    X = dfBX.drop(['quality'], axis=1)
    y = dfBX['quality']

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)  # 75% training and 25% test

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    warnings.filterwarnings('ignore')

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    F1_score = format(metrics.f1_score(y_test, y_pred, average='weighted'), '.3f')
    print("F1 score: ", F1_score)

    precision = format(metrics.precision_score(y_test, y_pred, average='weighted'), '.3f')
    print("Precision: ", precision)

    recall = format(metrics.recall_score(y_test, y_pred, average='weighted'), '.3f')
    print("Recall: ", recall)

    return F1_score, precision, recall
