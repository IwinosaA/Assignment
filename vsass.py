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
	#for row in csvreader:
	#print(row)

#Read the Body Column of the 'cnn_data_4_5.csv' file
content=['body']
cnn_data = pd.read_csv("cnn_data_4_5.csv", skipinitialspace=True, usecols=content)
print (cnn_data.body)