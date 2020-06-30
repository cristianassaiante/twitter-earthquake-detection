from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn import tree
import json
import numpy as np

def build_dataset(vectorizers, with_agency = False):
        
        x_text = []
        y = []
        
        with open('dataset', 'r') as f:
            for line in f:
                data = json.loads( line )
                
                if with_agency == False:
                    if data['y'] != 'd':
                        x_text.append( data['text'] )
                        y.append( 0 if data['y'] == 's' else 1 )
                else:
                    x_text.append( data['text'] )
                    y.append( 0 if data['y'] == 's' else 1 )
                    
        print("Dataset of size: %d" % len(y))
        y = np.array(y)
        x_vect = []
        for vectorizer in vectorizers:
            x_vect.append( vectorizer.fit_transform( x_text ).toarray() )
        return list(map(lambda w: train_test_split(w, y, test_size = 0.2), x_vect))
    
    
vec_clf_pairs = [ (CountVectorizer(binary = True, min_df = 0.05), BernoulliNB()),
                  (CountVectorizer(binary = True, min_df = 0.05), tree.DecisionTreeClassifier()),
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

