from twython import Twython
import json


with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

query = {
            #'q' : 'life',
            'count' : 10,
            'geocode' : '41.9109,12.4818,100km',
            'until' : '2020-05-12'
        }


for status in python_tweets.search(**query)['statuses']:
    print(status['text'])
   
