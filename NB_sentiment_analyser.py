# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd
import nltk
from nltk.corpus import stopwords

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "acpXXjd" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


class SentimentScales:
    # [ 'negative', 'neutral' , 'positive' ]
    scale_3 = [ 0, 1, 2 ]

    # [ 'negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive' ]
    scale_5 = [ 0, 1, 2, 3, 4 ]

class Review:
    def __init__(self, id, phrase, sentiment) :
        self.id = id
        self.phrase = phrase
        self.sentiment = sentiment
    
    def get_reviews(filename):
        reviews = []
        file = pd.read_csv(filename, index_col=0, delimiter='\t')

        for i, row in file.iterrows():
            if 'Sentiment' in file.columns:
                r = Review(i, row['Phrase'].split(), row['Sentiment'])
            else:
                r = Review(i, row['Phrase'].split(), -1)
            reviews.append(r)
        return reviews

class Preprocess:
    def __init__(self, reviews):
        self.reviews = reviews
        stop_list = stopwords.words('english')
        # Add some punctuation as the stopwords can be deleted
        stop_list.extend([ '``', '\'\'', '...', '--', '.', ',' , '\'', '-', ':', ';', '!', '?', '\'s', '`', '\'re', 'A.'])
        self.stop_words = set(stop_list)

    def preprocess_reviews(self):
        for review in self.reviews:
            # Tokenize the reviews using NLTK
            review.phrase = [r for r in review.phrase if not r.lower() in self.stop_words]
        return self.reviews



def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    reviews = Review.get_reviews('moviereviews/train.tsv')
    reviews_preprocessed = Preprocess(reviews).preprocess_reviews()
    
    
    f = open('debug.tsv', 'w')
    for review in reviews_preprocessed:
        f.write(str(review.phrase) + '\n')
    f.close()



    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = 0
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()

