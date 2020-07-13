import twint
import sqlite3
from datetime import datetime, timedelta
import sys
import json
import os
import numpy
import math
from twython import Twython
import googlemaps

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
"""
relevant_ids = []

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
                tweet_id = t['id']
                t = preprocess_tweet(t['tweet'])    
                t = 1 if is_tweet_about_earthquake(t) else 0
                n_relevant_tweets += t
                if t:
                    relevant_ids.append(tweet_id)

            print('[3] %d/%d relevant tweets'
                    % (n_relevant_tweets, len(tweets)))

            # 4. calculate p_occur and evaluate event detection
            p_occur = 1 - pow(pf, n_relevant_tweets)
            print('[4] p_occur = %f' % p_occur)
            
            




            
            print()
            os.unlink('tweets.json')
        else:
            break
        
        i += 1

    break

# 5. obtain location information about tweets

            
locations = []
import secrets
CONSUMER_KEY = secrets.CONSUMER_KEY
CONSUMER_SECRET = secrets.CONSUMER_SECRET
OAUTH_TOKEN = secrets.OAUTH_TOKEN
OAUTH_TOKEN_SECRET = secrets.OAUTH_TOKEN_SECRET
twitter = Twython(
    CONSUMER_KEY, CONSUMER_SECRET,
    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
           
req = 0
for t_id in relevant_ids:
    tweet = twitter.show_status(id = t_id)
                
    if tweet['coordinates'] != None:
        locations.append(tweet['coordinates']['coordinates'])
    req += 1
    if req >= 100:
        break
                #print (req)

# 6. calculate estimated location
#using Kalman Filters (Gaussian)
pos_estimate = (0.0 , 0.0 ) #the estimate is just the mean of the various locations
total = (0.0,0.0) #sum of all the locations, needed to compute the mean
total_square = (0.0, 0.0) #to calculate variance
sum_of_product = 0.0 #to calculate covariance
num_locations = 0 #number of locations considered so far
probability = 0 #probability that my estimation is correct
for pos in locations:
    total = (total[0] + pos[0], total[1] + pos[1])
    print (total)
    total_square = (total_square[0] + pos[0]**2, total_square[1] + pos[1]**2)
    sum_of_product += (pos[0]*pos[1])
    num_locations += 1
pos_estimate = (total[0]/num_locations, total[1]/num_locations)
variance_x = (total_square[0]/num_locations) - (pos_estimate[0]**2)
variance_y = (total_square[1]/num_locations) - (pos_estimate[1]**2)
covariance_x_y = (sum_of_product/num_locations) - (pos_estimate[0]*pos_estimate[1])

cov_matrix = numpy.array([[variance_x, covariance_x_y],[covariance_x_y, variance_y]])
cov_determinant = numpy.linalg.det(cov_matrix)


inverse_probability = (2*math.pi)* math.sqrt(cov_determinant)
           
 
            
            
print ("estimated position: " + str(pos_estimate[0]) + "; " + str(pos_estimate[1]))
if (inverse_probability > 0):
    print ("probability (density): " + str(1/inverse_probability))
    print("(if the variance is really small, the density may be greater than 1)")