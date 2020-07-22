from twython import Twython



def id_to_coord(twython_obj, tweet_id):
    tweet = twython_obj.show_status(id=tweet_id)
    if tweet['coordinates'] == None:
        return None
    else:
        return tweet['coordinates']['coordinates']



def twitter_object(key, secret, oauth, oauth_secret):
    twitter = Twython(
        key, secret,
        oauth, oauth_secret)
    return twitter
