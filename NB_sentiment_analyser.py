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


class Review:
    def __init__(self, id, phrase, sentiment) :
        self.id = id
        self.phrase = phrase
        self.sentiment = sentiment
        self.scale = value_5
    
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
            review.phrase = [r.lower() for r in review.phrase if not r.lower() in self.stop_words]
        return self.reviews

    def scale_3(self):
        self.scale = value_3
        for review in self.reviews:
            match review.sentiment:
                case 1: review.sentiment = 0
                case 2: review.sentiment = 1
                case 3: review.sentiment = 2
                case 4: review.sentiment = 2
        return self.reviews


value_3 = [ 'negative', 'neutral' , 'positive' ]
value_5 = [ 'negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive' ]

class Classifier:
    def __init__(self, reviews, scale):
        self.reviews = reviews
        self.prior = dict()
        self.scale = scale
        for i in range(len(self.scale)):
            self.prior[i] = 0 

    def prior_probability(self):
        self.total_reviews = len(self.reviews)
        for review in self.reviews:
                self.prior[review.sentiment] += 1
    
        for i in self.prior:
            self.prior[i] = self.prior[i] / self.total_reviews
        
        return self.prior
        
    def word_likelihood_calculator(self):
        word_count = dict()
    
        for i in range(len(self.scale)):  
            word_count[i] = dict()  
            for review in self.reviews:
                for word in review.phrase:
                    word_count[i][word] = 1

        for review in self.reviews:
            for word in review.phrase:
                word_count[review.sentiment][word] += 1
                
        
        likelihood_sum = dict()
        for sentiment in word_count:
            likelihood_sum[sentiment] =  sum(word_count[sentiment].values())

        self.word_likelihood = word_count
        for sentiment in word_count:
            for word in word_count[sentiment]:
                self.word_likelihood[sentiment][word] = word_count[sentiment][word] / (likelihood_sum[sentiment]*2)
        
        # for i in self.word_likelihood:
        #     self.word_likelihood[i] = sum(self.word_likelihood[i].values())

        return self.word_likelihood

    def review_sentiment_calculator(self):
        self.review_score = dict()
        for i in self.prior:
            self.review_score[i] = dict()
            
        for sentiment in self.review_score:
            for review in self.reviews:
                self.review_score[sentiment][review.id] = self.prior[sentiment]
                for word in review.phrase:
                    self.review_score[sentiment][review.id] = self.review_score[sentiment][review.id] * self.word_likelihood[sentiment][word]

        return self.review_score

    def review_sentiment_decider(self):
        self.review_predicted = dict()
        for review in self.reviews:
            temp = dict()
            for sentiment in self.review_score:
                temp[sentiment] = self.review_score[sentiment][review.id]
            self.review_predicted[review.id] = max(temp, key=temp.get)   
        return self.review_predicted
            
 
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
    if number_classes == 3: 
        reviews_preprocessed = Preprocess(reviews_preprocessed).scale_3()
        classifier = Classifier(reviews_preprocessed, value_3)
    else:
        classifier = Classifier(reviews_preprocessed, value_5)
    prior_prob = classifier.prior_probability()
    word_likelihoods = classifier.word_likelihood_calculator()
    reviews_scored = classifier.review_sentiment_calculator()
    reviews_predicted = classifier.review_sentiment_decider()

    f = open('debug.tsv', 'w')
    f.write(str(reviews_predicted) + '\n' )
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

