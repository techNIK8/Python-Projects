from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys  # gia na emfanizetai o plhrhs pinakas otan kanoume print
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

numpy.set_printoptions(threshold=sys.maxsize)

training_set = []
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# 1
# tokenize twn protasewn kai prosthhkh sth lista training_set
with open('onion-or-not-example.csv', 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        for text in line:
            words = word_tokenize(line[0])
            training_set.append(words)

# 2-3
# prosthkh twn leksewn sto arxeio processed_sentences.txt se periptwsh pou den yparxoun sth bash twn stopwords
output = open('processed_sentences.txt', 'w', encoding="utf8")
output.write("[")
for sentence in training_set[1::2]:
    output.write('"')
    for word in sentence:
        if word not in stop_words:
            output.write(ps.stem(word) + " ")
    output.write('",\n')
output.write("]")
output.close()

# 4
# Eyresh timwn tf-idf
f = open("processed_sentences.txt", "r", encoding="utf8")

# 1h lysh, petaei error gia memory otan vazw to plhres arxeio
vectorizer = TfidfVectorizer()  # Metatroph text se dianysmata, to opoio tha xrhsimopoihthei ws eisodos
vectors = vectorizer.fit_transform(f)  # Ekmathhsh vocabulary kai timwn idf, kai epistrofh timwn se mhtrwo.
feature_names = vectorizer.get_feature_names()  # antistoixhsh pinaka deiktwn se xarakthristika onomata
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df = df.sum(axis=0)
df.to_csv(r'Words_Tfidf_df.csv', index=True)


'''
# 2h lysh, typwnei tis times tf-idf se mia sthlh xwris tis lexeis
tfidf_vectorizer = TfidfVectorizer(use_idf=True)  # Metatroph text se dianysmata, to opoio tha xrhsimopoihthei ws eisodos
cv = CountVectorizer()
# Eisagwgh text
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(f)  # Ekmathhsh vocabulary kai timwn idf, kai epistrofh timwn se mhtrwo
# epistrofh prwtou dianysmatos
first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
# eisagwgh timwn tf-idf se pandas dataframe
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
print(df)
df.sort_values(by=["tfidf"], ascending=False)
df.to_csv(r'Final_dataset.csv', index=False)

'''




X = df.index
y = df.values

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=1)

X_train_tf = y

# If you have a GPU, you shouldn't care about AVX support,
# because most expensive ops will be dispatched on a GPU device (unless explicitly set not to).
# In this case, you can simply ignore this warning by
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = keras.Sequential([
    layers.Dense(10, input_dim=X_train_tf.shape[0], activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

m = keras.metrics.Precision()
# print('Precision result: ', m.result().numpy())

# los, precision, recall = model.evaluate(X_train_tf, y_test)
