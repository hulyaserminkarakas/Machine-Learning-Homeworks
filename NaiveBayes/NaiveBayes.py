import numpy as np;
from sklearn.feature_extraction.text import TfidfVectorizer
tweets = np.load("train_tweets.npy")
validationTweets = np.load("validation_tweets.npy")

negative = []
neutral = []
positive = []

cv = TfidfVectorizer(ngram_range=(1,2),max_df=0.8,min_df=2) #normalized count vectorizer
mat = cv.fit_transform(tweets[:,1]).toarray()

for i,t in zip(mat,tweets[:,0]):
    if t == b'0':
        negative.append(i)
    if t == b'2':
        neutral.append(i)
    if t == b'4':
        positive.append(i)

negative=np.array(negative)
neutral=np.array(neutral)
positive=np.array(positive)

negativeCP = np.log(len(negative) / len(tweets))  #Class Prior
positiveCP = np.log(len(positive) / len(tweets))
neutralCP = np.log(len(neutral) / len(tweets))

negativeSum = negative.sum()+ negative.shape[1]  #for smooting we add the column number of negative array
neutralSum = neutral.sum()+ neutral.shape[1]
positiveSum = positive.sum()+ positive.shape[1]

negativeLH = list() #likelyhood
neutralLH = list() #likelyhood
positiveLH = list() #likelyhood
for i in range(negative.shape[1]):
    negativeLH.append(np.log((negative[:,i].sum()+1.0) / negativeSum))
    neutralLH.append(np.log((neutral[:, i].sum() + 1.0) / neutralSum))
    positiveLH.append(np.log((positive[:, i].sum() + 1.0 )/ positiveSum))


val=cv.transform(validationTweets[:,1]).toarray()
labels=validationTweets[:,0]

predict = list()
for test in val:


    negPosterior =(negativeLH * test).sum() + negativeCP
    neutPosterior = (neutralLH * test).sum() + neutralCP
    posPosterior = (positiveLH * test).sum() + positiveCP
    prediction=np.max([negPosterior, neutPosterior, posPosterior])

    if(prediction == negPosterior):
        predict.append(b'0')
    if (prediction == neutPosterior):
        predict.append(b'2')
    if (prediction == posPosterior):
        predict.append(b'4')

count = 0
for i,j in zip(labels,predict):
    if(i == j):
        count += 1

accuracy = (count / len(labels)) *100
print(accuracy)
