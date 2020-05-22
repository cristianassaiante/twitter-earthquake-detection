import sqlite3
from twython import Twython
import json


def tweets_from_location(lat, lon, ts):
    query = {
            'count' : 100,
            'geocode' : '%s,%s,5km'  % (lat, lon),
        }




with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

