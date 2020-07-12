import twint
import sqlite3
from datetime import datetime, timedelta
import sys
import json
import os

from classifier import is_tweet_about_earthquake, preprocess_tweet


"""
    RETRIEVE EARTHQUAKES
"""
connection = sqlite3.connect('earthquakes.db')
c = connection.cursor()
assert os.path.isfile("dataset"), "Dataset file does not exists"
result_set = c.execute('''select id, latS, lonE, timestamp
                          from earthquakes
                          order by felt
                          desc''')

"""
    PARAMETERS
"""
pf = 0.35 
delta = 5   # minutes

"""
    ALGORITHM 2
"""
# start by using known earthquakes for experiments
for _, latS, lonE, ts in result_set:

    i = 0
    while True:   # simulate the "every s seconds" described in the paper
        # retrieve tweets
        print('[%d minutes from earthquake]' % (i * delta))

        since = datetime.fromtimestamp(ts) + timedelta(minutes = i * delta)
        until = since + timedelta(minutes = delta)
        
        c = twint.Config()
        c.Search = "earthquake"
        c.Since = str(since)
        c.Until = str(until)
        c.Geo = '%lf,%lf,50km' % (latS, lonE)
        c.Output = 'tweets.json'
        c.Store_json = True
        c.Hide_output = True
        
        twint.run.Search(c)

        # work with set of retrieved tweets
        if os.path.exists('tweets.json'):
            tweets = []
            for tweet in open('tweets.json'):
                tweets.append(json.loads(tweet))

            print('[2] %d tweets retrieved' % len(tweets))

            # 3. obtain features and do classification
            n_relevant_tweets = 0
            for t in tweets:
                t = preprocess_tweet(t['tweet'])
                t = 1 if is_tweet_about_earthquake(t) else 0
                n_relevant_tweets += t

            print('[3] %d/%d relevant tweets'
                    % (n_relevant_tweets, len(tweets)))

            # 4. calculate p_occur and evaluate event detection
            p_occur = 1 - pow(pf, n_relevant_tweets)
            print('[4] p_occur = %f' % p_occur)
            
            # 5. obtain location information about tweets

            # 6. calculate estimated location
                
            print()
            os.unlink('tweets.json')
        else:
            break
        
        i += 1

    break

