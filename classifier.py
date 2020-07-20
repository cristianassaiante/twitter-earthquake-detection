from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn import tree
import json
import numpy as np
import nltk
nltk.download('stopwords', quiet = True)
nltk.download('punkt',     quiet = True)
nltk.download('wordnet',   quiet = True)


model      = None
vectorizer = None
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer    = nltk.stem.PorterStemmer()

keywords   = list(map(stemmer.stem, ['earthquake', 'shaking']))

# remove punctuation, lemmatize and stem
def preprocess_tweet(text):
    tweet = nltk.wordpunct_tokenize(text) 
    tweet = [w.lower() for w in tweet if w.isalpha()]
    tweet = map(lemmatizer.lemmatize, tweet)
    tweet = list(map(stemmer.stem, tweet) )
    context = []
    
    for kword in keywords:
        if kword in tweet:
            idx = tweet.index(kword)
            if idx == 0:
                context.append('')
            else:
                context.append(tweet[idx - 1])
            if idx == len(tweet) - 1:
                context.append('')
            else:
                context.append(tweet[idx + 1])
            break
    else:
        raise RuntimeError('Tweet "%s" does not contain any keyword' % text)
    size = len(tweet)
    tweet = ' '.join( map(stemmer.stem, tweet) )
    return tweet, context, size

# creates a dataset where each entry is vectorized_tweet + 
# [index in vectorizer vocabulary of context before keyword] + 
# [index in vectorizer vocabulary of context after keyword] +
# number of terms in lemmatized tweet
def build_dataset(vectorizers, with_unsure = False, test_ratio = 0.2):
        x_text    = []
        x_context = []
        x_length  = []
        y         = []
        
        with open('dataset', 'r') as f:
            for line in f:
                data = json.loads( line )
                text = data['text']
 
                text, context, length = preprocess_tweet(text)

                if with_unsure == False:
                    if data['y'] != 'd':
                        x_text.append( text )
                        y.append( 0 if data['y'] == 's' else 1 )
                else:
                    x_text.append( text )
                    y.append( 0 if data['y'] == 's' else 1 )
                x_length.append( [length, ] )
                x_context.append( context )
        x_length = np.array( x_length )
        y = np.array(y).astype(np.uint8)
        
        x_vect   = []
        for vectorizer in vectorizers:
            x_context_vec = np.array( x_context )
            
            x1 = vectorizer.fit_transform( x_text ).toarray()
            
            for i in range(len(x_context)):
                x_context_vec[i] = np.array(list( map( lambda x: vectorizer.vocabulary_.get(x, -1), x_context_vec[i] ) ))
            
            x = np.hstack( [x1, x_context_vec, x_length] ).astype(np.float64)
            x_vect.append( x )
        #x_vect = x_vect.astype(np.float64)
        return list(map(lambda w: train_test_split(w, y, test_size = test_ratio), x_vect))
    

def is_tweet_about_earthquake(text):
    global vectorizer, model
    # Model and vectorizer do not exist yet, create them
    if model == None:
        vectorizer, model = CountVectorizer(binary = True, min_df = 0.05), tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
        x_train, x_test, y_train, y_test = build_dataset( [vectorizer], with_unsure = True, test_ratio = 1)[0]
        model.fit(x_train, y_train)
        
        # Export the decision tree in .dot format, `dot -Tpng model.dot > model.png; display model.png` to show it
        dot_str = tree.export_graphviz(model, feature_names = vectorizer.get_feature_names() + ['context_before', 'context_after', 'tweet_length'], impurity = False)
        
        with open("model.dot", "w") as f:
            f.write(dot_str)
    
    
    text, context, length = preprocess_tweet(text)
    x_context = list( map( lambda x: vectorizer.vocabulary_.get(x, -1), context ) )
    
    x = np.concatenate( (vectorizer.transform( [text] ).toarray()[0], x_context, [length]) )
    
    return model.predict( [ x ] )[0] == True
    
    
if __name__ == '__main__':
    vec_clf_pairs = [
                        (CountVectorizer(binary = True, min_df = 0.05), BernoulliNB()),
                        (CountVectorizer(binary = True, min_df = 0.05), tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)),
                        (TfidfVectorizer(min_df = 0.05),                svm.SVC())
                    ]

    datasets = build_dataset( (x[0] for x in vec_clf_pairs),
                            with_unsure = True)
    for (x_train, x_test, y_train, y_test), clf in zip(datasets, [w[1] for w in vec_clf_pairs]):
        #print(x_train.shape, y_train.shape, x_)
        clf.fit( x_train.astype(np.float64), y_train )
        print("%-30.30s accuracy: %.3lf precision: %.3lf" % ( clf.__class__.__name__,
                                                            clf.score( x_test, y_test ),
                                                            precision_score( y_test, clf.predict( x_test ))
                                                            ))

