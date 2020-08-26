# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:30:39 2019

@author: khush
"""

# Natural Language Processing
    
# Importing the libraries

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('air.csv', encoding='ISO-8859–1')

# Cleaning the texts
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


for i in range(0, 10898):
    
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    #'Not' is replaced by 'Nots' so that it will not be detected by stopwords.
    text=re.sub("not","nots",text);
    text = text.lower() 
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)


# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 2:5]


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = i)


from scipy.sparse import lil_matrix
# Note that this classifier can throw up errors when handling sparse matrices.

X_train = lil_matrix(X_train).toarray()
y_train = lil_matrix(y_train).toarray()
X_test = lil_matrix(X_test).toarray()





#Comparison models

from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from skmultilearn.adapt import MLkNN
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

classifiers=[
    (LabelPowerset(GaussianNB()),"GNB"),
    (LabelPowerset(SVC(kernel = 'rbf', random_state = 0)),"SVC"),
    (LabelPowerset(DecisionTreeClassifier(random_state = 0)),"DTC"),
    (MLkNN(k=32),"MLKNN"),
    (LabelPowerset(RandomForestClassifier(n_estimators=90, max_depth=70,random_state=0)),"RFC"),
    (LabelPowerset(LogisticRegression()),"LR"),
    (LabelPowerset(KNeighborsClassifier(5)),"KNC"),
    (LabelPowerset(AdaBoostClassifier()),"ADC"),
]

print("ACCURACIES WHEN USING COUNT VECTORIZER:")
#Accuracy scores of different models
score_ , names = [] , [] 
for model,name in classifiers:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score_.append(accuracy_score(y_test,y_pred)*100)
    names.append(name)
    
    
    
    
    
#Graph to show the accuracy of different models
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
A = score_[:]
plt.plot(A)
for i, label in enumerate(names):
    plt.text(i,A[i], label) 
plt.show() 




#TFIDF
# word level tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=200)
tfidf_vect.fit(corpus)
X_tfidf =  tfidf_vect.transform(corpus)
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
y = dataset.iloc[:, 2:5]
xtrain_tfidf, xtest_tfidf, ytrain_tfidf, ytest_tfidf = train_test_split(X_tfidf, y, test_size = 0.20, random_state = 68)

 
#Comparison models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import accuracy_score

classifiers=[
    (LabelPowerset(GaussianNB()),"GNB"),
    (LabelPowerset(SVC(kernel = 'rbf', random_state = 0)),"SVC"),
    (LabelPowerset(DecisionTreeClassifier(random_state = 0)),"DTC"),
    (LabelPowerset(RandomForestClassifier(n_estimators=80, max_depth=100,random_state=0)),"RFC"),
    (LabelPowerset(LogisticRegression()),"LR"),
    (LabelPowerset(KNeighborsClassifier(5)),"KNC"),
    (LabelPowerset(AdaBoostClassifier()),"ADC"),
]


print("ACCURACIES WHEN USING TFIDF:")
#Accuracy scores of different models
score_ , names = [] , []
for model,name in classifiers:
    model.fit(xtrain_tfidf, ytrain_tfidf)
    ypred_tfidf = model.predict(xtest_tfidf)
    score_.append(accuracy_score(ytest_tfidf,ypred_tfidf)*100)
    names.append(name)
    
#Graph to show the accuracy of different models
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
A = score_[:]
plt.plot(A)
for i, label in enumerate(names):
    plt.text(i,A[i], label) 
plt.show()
    


# Bar graph to count number of texts per category
df_toxic = dataset.drop(['id', 'text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats
print(categories)

df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of texts per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Category', fontsize=12)

# Creating the wordcloud
from wordcloud import WordCloud,STOPWORDS
comment_words = ' '

stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in corpus: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '

  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

Y_test = y_test.values.argmax(axis=1)
Y_pred = y_pred.toarray().argmax(axis=1)
categories = np.array(categories)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, Y_pred, classes=np.array(categories),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, Y_pred, classes=categories, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print(categories)