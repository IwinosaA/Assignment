'''
Name: Iwinosa Agbonlahor
Class: EEGR 565 Machine Learning Applications
Assignment 1:
This is a Program that reads the content of a csv file and plot a histogram of the words,
indicating which word is related to pandemic and the frequency of its occurence.

Link to shared folder: 
'''

# Here we import the necessary libraries, objects and function for the application
import pandas as pd
import numpy as np
import csv
from nltk.corpus import names
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer


# display first five rows of the file
with open("cnn_data_4_5.csv", 'r') as file:
	csvreader = csv.reader(file)
	for row in csvreader:
	print(row [:5])


#Read the Body Column of the 'cnn_data_4_5.csv' file
content=['body']
cnn_data = pd.read_csv("cnn_data_4_5.csv", skipinitialspace=True, usecols=content)
# print (cnn_data.body)


# This Vectorizes the words in the body section and sorts it in order of freq
cv=CountVectorizer(stop_words="english", max_features=500)
bag_of_words=cv.fit_transform(cnn_data.body)


# Plot a histogram of the most frequent words in the dataset
import matplotlib.pyplot as plt
keywords=[]
freqs=[]
for word, count in words_freq:
    keywords.append(word)
    freqs.append(count)
plt.bar(np.arange(10), freqs[:10], align='center')
plt.xticks(np.arange(10), keywords[:10])
plt.ylabel('Frequency')
plt.title('Top 10 Keywords')
plt.show()

selected_tokens = list(filter(lambda freq: c[freq] > 1000, token_list))

#Test if a token is a word
def letters_only(astr):
    return astr.isalpha()
#Remove names from words and perform word lemmatization
cleaned = []
all_names = set(x.lower() for x in names.words())
lemmatizer=WordNetLemmatizer()
for post in cnn_data.body[:250]:
    cleaned.extend(list(lemmatizer.lemmatize(word.lower()) for word in post.split()
                if letters_only(word) and word.lower() not in all_names)) 
print(cv.get_feature_names())

sum_words=cleaned.sum(axis=0)
words_freq=[(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x:x[1], reverse=True)
for word, count in words_freq:
    print(word+":",count)
 
