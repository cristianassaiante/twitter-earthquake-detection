import twint
import sqlite3
from datetime import datetime, timedelta
import sys
import json
import os

connection = sqlite3.connect('earthquakes.db')
c = connection.cursor()


DELTA = 2 # hours from start of earthquake 

ROWS, COLS = map(int, os.popen('stty size', 'r').read().split()) # rows and cols of console

assert len(sys.argv) == 3 and sys.argv[1] == '-id' and 0 <= int(sys.argv[2]) <= 2, 'No'
userid = int(sys.argv[2])

assert os.path.isfile("dataset"), "Dataset file does not exists"

blocklist = ['Earthquake Map',
             'Quake Reports',
             'Sismo Mapa',
             'Earthquake Alerts',
             'Every Earthquake',
             'SF QuakeBot',
             'EMSC',
             'CA Earthquake Bot',
             'CA/NV Earthquakes',
             'Southern CA Quakes',
             'San Diego Earthquake',
             'Large Quakes SF'] # inclusivity 100


result_set = c.execute('''select id, latS, lonE, timestamp
                          from earthquakes
                          where timestamp %% 3 = %d
                          order by felt
                          desc''' % userid)

for id, latS, lonE, ts in result_set:
    since = datetime.fromtimestamp(ts)
    until = since + timedelta(hours = DELTA)
    
    c = twint.Config()
    c.Since = str(since)
    c.Until = str(until)
    c.Geo = '%lf,%lf,1km' % (latS, lonE)
    c.Output = 'tweets.json'
    c.Store_json = True
    c.Hide_output = True
    
    twint.run.Search(c)
    
    if os.path.exists('tweets.json'):
        for line in open('tweets.json'):
            tweet = json.loads(line)
            text  = tweet['tweet']
            tid   = tweet['id']
            hid   = id
            user  = tweet['name']
            
            if user in blocklist: continue
            
            sys.stdout.write('\b' * (ROWS * COLS))
            x = user + "\n" + text
            xrows = 1 + text.count("\n") + round(len(text) / float(COLS) + 0.5)
            
            sys.stdout.write(x)
            sys.stdout.write("\n" * (ROWS - xrows - 3))
            
            sys.stdout.write("a - tweet is about earthquakes\ns - tweet is not about earthquakes\nd - unsure or from some agency\n")
            sys.stdout.flush()
            
            y = None
            while y not in ['a', 's', 'd']:
                y = input()
            
            output = json.dumps( {'text' : text, 'y' : y, 'tid' : tid, 'hid' : hid} )
            
            with open("dataset", 'a') as f:
                f.write(output + "\n")
            os.system("clear")
            
        os.unlink('tweets.json')
            
    


