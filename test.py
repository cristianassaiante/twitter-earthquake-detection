import classifier


tweet = "earthquake, please help i'm too young to die in such an unfortunate circumstance"

print(classifier.preprocess_tweet(tweet))

print(classifier.is_tweet_about_earthquake(tweet))
