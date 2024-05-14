File onion-or-not.csv contains two columns. 
The first column includes headlines from fake news while the second column informs us 
whether they were published on the well-known humorous website theonion.com or not. 
The goal is to try to guess the information in the second column using a neural network. 
To transform the article titles to create the registry that will be given as input to the 
model being trained the procedure below is followed:
1. Split the titles into words, creating a vector of words.
2. From the words their endings are removed, keeping only their subject (stemming).
3. Remove from collection those words that are quite common and offer no information (stopwords removal).
4. The remaining words will be weighted by tf-idf.
5. Combine vectors to produce the final register.
After generating the registry, we split it into a training-test dataset with a ratio of 75%-25%. Next, a neural network is trained and its performance is measured using the f1 score, precision and recall metrics.