# -- coding: utf-8 --
"""
Created on Mon Sep 30 22:57:46 2019


"""

import tweepy
import csv
import pandas as pd

import re

####input your credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('air1.csv', 'w')
#Use csv Writer
csvWriter = csv.writer(csvFile)
c=["air+pollution","soil+pollution","water+pollution"]
for tweet in tweepy.Cursor(api.search,q=c[0],count=1000,lang="en",since="2016-04-03",tweet_mode="extended").items(10000):
        # print (tweet.full_text)
    # print("printed")
    if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
        csvWriter.writerow([tweet.created_at,' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split())])
        # print(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split()))
print('air done')
csvFile = open('soil2.csv', 'w')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q=c[1],count=1000,lang="en",tweet_mode="extended").items(50000):
        # print (tweet.full_text)
        # print("printed")
    if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
        csvWriter.writerow([tweet.created_at,' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split())])
        # print(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split()))
csvFile = open('water1.csv', 'w')
#Use csv Writer
csvWriter = csv.writer(csvFile)
print('soil done')
for tweet in tweepy.Cursor(api.search,q=c[2],count=1000,lang="en",since="2017-04-03",tweet_mode="extended").items(10000):
        # print (tweet.full_text)
        # print("printed")
   if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
       csvWriter.writerow([tweet.created_at,' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split())])
        # print(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.full_text).split()))s