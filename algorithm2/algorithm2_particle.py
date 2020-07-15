import twint
import sqlite3
from datetime import datetime, timedelta
import sys
import json
import os
import numpy
import math
from twython import Twython

from classifier import is_tweet_about_earthquake, preprocess_tweet
import secrets

from contextlib import redirect_stdout
import io
import random


"""
    Initialize Twython
"""
CONSUMER_KEY = secrets.CONSUMER_KEY
CONSUMER_SECRET = secrets.CONSUMER_SECRET
OAUTH_TOKEN = secrets.OAUTH_TOKEN
OAUTH_TOKEN_SECRET = secrets.OAUTH_TOKEN_SECRET
twitter = Twython(
    CONSUMER_KEY, CONSUMER_SECRET,
    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
tot_req = 0

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

"""
VARIABLES
"""
current_estimate = None
previous_estimate = None
old_variance_x = None
old_variance_y = None
old_covariance = None
old_cov_matrix = None
old_cov_determinant = None

"""
    ALGORITHM 2
"""
# start by using known earthquakes for experiments
for _, latS, lonE, ts in result_set:

    i = 0
    while True:   # simulate the "every s seconds" described in the paper
        if i >=20: #just to prevent it from running forever
            break
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
            relevant_ids = []
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
            
            # 5. obtain location information about tweets
            locations = []
            skip = False
            req = 0
            for t_id in relevant_ids:
                tweet = twitter.show_status(id = t_id)
                            
                if tweet['coordinates'] != None:
                    locations.append(tweet['coordinates']['coordinates'])
                req += 1
                tot_req +=1
                if (req >= 100 or tot_req >= 800):
                    break
            if len(locations) < 5:
                skip = True

            if skip:    
                i += 1
                continue
                
            

            # 6. calculate estimated location
            # Algorithm 1
            # 6.2. Give a weight to tweets
            # Idea for weight: compute average number of tweets in that area per unit of time
            # More tweets in that area means less weight. Add 1 to avoid giving infinte weight.
            # Will adjust the parameters and an average shoud be done, but for now I will just search once
            #searching for tweets takes way too much time, but it could easily be reduced by doing it in parallel
            weights = []
            zone_tweets = []
            total_tweets_for_weight = 0
            for tweet in locations:
                c = twint.Config()
                c.Since = "2020-06-01 21:30:00" ##these will be adjusted, the idea is to take the same time but on a normal day (or more normal days)
                c.Until = "2020-06-01 21:50:00"
                geostring = str(tweet[0]) + ", " + str(tweet[1]) + ", 50km" 
                c.Geo = geostring
                c.Hide_output = True
                c.Count = True
                f = io.StringIO()
                with redirect_stdout(f):
                    twint.run.Search(c)
                s = f.getvalue()
                last = s.rfind('T')
                num_of_tweets = int(s[37:last]) #please don't judge
                zone_tweets.append(num_of_tweets+1)
                total_tweets_for_weight += num_of_tweets+1

            for z in zone_tweets:
                weights.append(1-(z/total_tweets_for_weight)) #make so that they weigh less if they have more tweets in their area
            sumCheck = 0
            for w in weights:
                sumCheck += w
            
            #and then normalize
            for w in range(0, len(weights)):
                weights[w] = weights[w]/sumCheck
            
            #6.2.5. If it is the first iteration, just compute the average and continue
            
            if(previous_estimate == None):
                sum_of_x = 0.0
                sum_of_y = 0.0
                square_sum_of_x = 0.0 #for variance
                square_sum_of_y = 0.0
                sum_of_product = 0.0 #for covariance
               
                for l in range(0,len(locations)):
                    sum_of_x += (locations[l][0]*weights[l])
                    sum_of_y += (locations[l][1]*weights[l])
                    square_sum_of_x += ((locations[l][0]**2)*weights[l])
                    square_sum_of_y += ((locations[l][1]**2)*weights[l])
                    sum_of_product += ((locations[l][0]*locations[l][1])*weights[l])

                current_estimate = (sum_of_x, sum_of_y)
                previous_estimate = (sum_of_x, sum_of_y)
                old_variance_x = square_sum_of_x - (previous_estimate[0]**2)
                old_variance_y = square_sum_of_y - (previous_estimate[1]**2)
                old_covariance = sum_of_product - (previous_estimate[0]*previous_estimate[1])
                old_cov_matrix = numpy.array([[old_variance_x, old_covariance],[old_covariance, old_variance_y]])
                old_cov_determinant = numpy.linalg.det(old_cov_matrix)
                print ("estimated position: " + str(current_estimate[0]) + "; " + str(current_estimate[1]))
                #skip = True
                i += 1
                continue
            #6.3 re-sample
            # make a new list of tweets, it will not include every tweet but every tweet will be taken with a chance proportional to its weight
            # the same tweet could be taken more than once. In the new list every tweet will have the same weight
            
            else:
                resample_locations = []
                for l in range(0,len(locations)):
                    a = random.uniform(0,1)
                    pick = 0
                    sum_of_weights = 0
                    for j in range (0,len(locations)):
                        sum_of_weights += weights[j]
                        if sum_of_weights >= a:
                            pick = j
                            break
                        else:
                            
                            continue

                    resample_locations.append(locations[pick])

                #6.4 re-weigh
                # before re-weighting I should do the prediction of the next state, but I am not sure how it is done in the paper
                # so I am just going to base the new weight on the difference with the previous estimation
                # the more they are close to the previous estimation, the more they weigh
                # similarity is computed using a 2D Gaussian, the mean is the previous estimation, the variances are 
                inverse_old_matrix = numpy.linalg.inv(old_cov_matrix)
                likelihood_from_previous = []
                sum_of_likelihoods = 0
                old_x = previous_estimate[0]
                old_y = previous_estimate[1]
                root_matrix_det = math.sqrt(old_cov_determinant)
                for l in resample_locations:
                    dif = numpy.array([l[0] - old_x, l[1] - old_y])
                    dif_tr = dif.copy()
                    dif_tr.transpose()
                    mult_1 = numpy.matmul(dif_tr,inverse_old_matrix)
                    mult_2 = numpy.matmul(mult_1, dif)
                    likelihood = (1/(2*math.pi*root_matrix_det))*math.exp((-0.5)*int(mult_2))
                    sum_of_likelihoods += likelihood
                    likelihood_from_previous.append(likelihood)
                new_weights = []
                if sum_of_likelihoods == 0.0:
                    i +=1
                    continue
                for l in range(0,len(resample_locations)):
                    new_weights.append(likelihood_from_previous[l]/sum_of_likelihoods)
                #6.5 new estimate
                # with the new weights I calculate the new estimate and the values I will need for the next iteration
                sum_of_x = 0.0
                sum_of_y = 0.0
                square_sum_of_x = 0.0 #for variance
                square_sum_of_y = 0.0
                sum_of_product = 0.0 #for covariance
               
                for l in range(0,len(resample_locations)):
                    sum_of_x += (resample_locations[l][0]*new_weights[l])
                    sum_of_y += (resample_locations[l][1]*new_weights[l])
                    square_sum_of_x += ((resample_locations[l][0]**2)*new_weights[l])
                    square_sum_of_y += ((resample_locations[l][1]**2)*new_weights[l])
                    sum_of_product += ((resample_locations[l][0]*resample_locations[l][1])*new_weights[l])

                estimate = (sum_of_x, sum_of_y)
                variance_x = square_sum_of_x - (estimate[0]**2)
                variance_y = square_sum_of_y - (estimate[1]**2)
                covariance = sum_of_product - (estimate[0]*estimate[1])
                cov_matrix = numpy.array([[variance_x, covariance],[covariance, variance_y]])
                cov_determinant = numpy.linalg.det(cov_matrix)
                try:
                    numpy.linalg.inv(cov_matrix)
                except:
                    skip = True
                if(skip):
                    i +=1
                    continue
                current_estimate = estimate
                previous_estimate = estimate
                old_variance_x = variance_x
                old_variance_y = variance_y
                old_covariance = covariance
                old_cov_matrix = cov_matrix
                old_cov_determinant = cov_determinant

                print ("estimated position: " + str(current_estimate[0]) + "; " + str(current_estimate[1]))



            
            print()
            os.unlink('tweets.json')
        else:
            break
        
        i += 1  #possible mistake: in some cases I skip the rest of the loop iteration, so that it won't reach this point. 
                #I made sure to increase i in those cases, but a mistake is still possible. If there is a mistake you can't understand, it is possibly because of this

    break



print("if the estimation suddenly changes it is probably because there is a lack of tweets")
print("it is safer to take as the best estimate the one before the sudden change")
# in general will perform poorly when there are close to no tweets available