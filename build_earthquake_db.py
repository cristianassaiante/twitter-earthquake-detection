import requests
import json
import sqlite3
from datetime import date

connection = sqlite3.connect("earthquakes.db")
c = connection.cursor()
c.execute("""
    create table if not exists earthquakes (
                                                id text primary key, latS decimal(3, 4), lonE decimal(3, 4), timestamp integer, magnitude decimal(1, 2), 
                                                magnitude_type text, alert text nullable, felt integer nullable
                                            )
    """)

# https://earthquake.usgs.gov/fdsnws/event/1/

data = []

for year in range(2019, 2021):
    
    # yyyy-mm-dd because the world is a bad place and we all deserve to die
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=%d-01-01&endtime=%d-01-01&minmagnitude=3" % (year, year + 1)

    response = requests.get(url)
    
    for earthquake in response.json()['features']:
        if earthquake['properties']['type'] == 'earthquake': # there are a few nuclear explosions, what you gonna do
            try:
                date.fromtimestamp(earthquake['properties']['time'])
            except:  # timestamp in milliseconds
                data.append( (
                                earthquake['id'],                         # Unique id
                                earthquake['geometry']['coordinates'][1], # South
                                earthquake['geometry']['coordinates'][0], # East
                                earthquake['properties']['time'] // 1000, # Timestamp
                                earthquake['properties']['mag'],          # Magnitude
                                earthquake['properties']['magType'],      # Magnitude type (?)
                                earthquake['properties']['alert'],        # Alert (green yellow orange red)
                                earthquake['properties']['felt']          # How many people reported it
                            )
                            )
            else:   # timestamp in seconds
                data.append( (
                                earthquake['id'],                         # Unique id
                                earthquake['geometry']['coordinates'][1], # South
                                earthquake['geometry']['coordinates'][0], # East
                                earthquake['properties']['time'],         # Timestamp
                                earthquake['properties']['mag'],          # Magnitude
                                earthquake['properties']['magType'],      # Magnitude type (?)
                                earthquake['properties']['alert'],        # Alert (green yellow orange red)
                                earthquake['properties']['felt']          # How many people reported it
                            )
                            )
            
    c.executemany('insert into earthquakes values (?, ?, ?, ? ,?, ?, ?, ?)', data)
    print("Added %d events" % len(data))
    data.clear()
    
                     
connection.commit()
connection.close()


