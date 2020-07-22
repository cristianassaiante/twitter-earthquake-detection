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

# remove punctuation, lemmatize and stem
def preprocess_tweet(text):
    tweet = nltk.wordpunct_tokenize(text) 
    tweet = [w.lower() for w in tweet if w.isalpha()]
    tweet = map(lemmatizer.lemmatize, tweet)
    tweet = ' '.join( map(stemmer.stem, tweet) )
    return tweet


def build_dataset(vectorizers, with_agency = False, test_ratio = 0.2):
        x_text = []
        y      = []
        
        with open('dataset.json', 'r') as f:
            for line in f:
                data = json.loads( line )
                text = data['text']
                text = preprocess_tweet(text)
                if with_agency == False:
                    if data['y'] != 'd':
                        x_text.append( text )
                        y.append( 0 if data['y'] == 's' else 1 )
                else:
                    x_text.append( text )
                    y.append( 0 if data['y'] == 's' else 1 )
                    
                    
        y = np.array(y)
        x_vect = []
        for vectorizer in vectorizers:
            x_vect.append( vectorizer.fit_transform( x_text ).toarray() )
        return list(map(lambda w: train_test_split(w, y, test_size = test_ratio), x_vect))
    

def is_tweet_about_earthquake(text):
    global vectorizer, model
    text = preprocess_tweet(text)
    
    # Model and vectorizer do not exist yet, create them
    if model == None:
        vectorizer, model = CountVectorizer(binary = True, min_df = 0.05), tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
        x_train, x_test, y_train, y_test = build_dataset( [vectorizer], with_agency = True, test_ratio = 1)[0]
        model.fit(x_train, y_train)
        
        # Export the decision tree in .dot format, `dot -Tpng model.dot > model.png; display model.png` to show it
        dot_str = tree.export_graphviz(model, feature_names = vectorizer.get_feature_names(), impurity = False)
        
        with open("model.dot", "w") as f:
            f.write(dot_str)
        
    return model.predict( vectorizer.transform( [text] ).toarray() )[0] == True
    
    
if __name__ == '__main__':
    vec_clf_pairs = [
                        (CountVectorizer(binary = True, min_df = 0.05), BernoulliNB()),
                        (CountVectorizer(binary = True, min_df = 0.05), tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)),
                        (TfidfVectorizer(min_df = 0.05),                svm.SVC())
                    ]

    datasets = build_dataset( (x[0] for x in vec_clf_pairs),
                            with_agency = True)

    for (x_train, x_test, y_train, y_test), clf in zip(datasets, [w[1] for w in vec_clf_pairs]):
        clf.fit( x_train, y_train )
        print("%-30.30s accuracy: %.3lf precision: %.3lf" % ( clf.__class__.__name__,
                                                            clf.score( x_test, y_test ),
                                                            precision_score( y_test, clf.predict( x_test ))
                                                            ))

