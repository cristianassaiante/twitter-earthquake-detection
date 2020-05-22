import sqlite3
from twython import Twython
import json
import time
from datetime import date

conn = sqlite3.connect('earthquakes.db')
c = conn.cursor()

earthquakes = c.execute("select * from earthquakes where timestamp > %d order by timestamp desc" % (int(time.time()) - 7 * 24 * 60 * 60)).fetchall()


def tweets_from_location(lat, lon, ts):
    query = {
            'count' : 100,
            'geocode' : '%lf,%lf,5km'  % (lat, lon),
            'until' : date.fromtimestamp(ts).isoformat()
        }
    
    tweets = python_tweets.search(**query)['statuses']
    length = len(tweets)
    max_display = 10
    
    for idx, status in enumerate(tweets):
        print("Tweet #%d" % idx)
        print(status['text'])
        print()
        if idx == max_display - 1 and length != max_display:
            print("... and other %d tweets\n\n" % (length - max_display))
            break
        

with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)

python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

print("Got %d earthquakes" % len(earthquakes))



for e in earthquakes:
    try:
        date.fromtimestamp(e[3])
    except:
        assert False, 'Wrong timestamp'
    print("Processing earthquake %s (%s)" % ( e[0], date.fromtimestamp(e[3]).isoformat() ) )
    tweets_from_location(e[1], e[2], e[3])
    
    
