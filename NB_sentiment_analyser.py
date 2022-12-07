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
USER_ID = "acd20xh" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

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
        stop_list.extend([ '``', '\'\'', '...', '--', '.', ',' , '\'', '-', ':', ';', '\'s', '`', '\'re', 'A.'])
        self.stop_words = set(stop_list)

    def preprocess_reviews(self):
        for review in self.reviews:
            # Tokenize the reviews using NLTK
            review.phrase = [r.lower() for r in review.phrase if not r.lower() in self.stop_words]
        return self.reviews

    def scale_3(self):
        for review in self.reviews:
            match review.sentiment:
                case 1: review.sentiment = 0
                case 2: review.sentiment = 1
                case 3: review.sentiment = 2
                case 4: review.sentiment = 2
        return self.reviews


class Classifier:
    def __init__(self, reviews, scale):
        self.reviews = reviews
        self.prior = dict()
        self.scale = scale
        for i in range(self.scale):
            self.prior[i] = 0 

    def prior_probability(self):
        self.total_reviews = len(self.reviews)
        for review in self.reviews:
                self.prior[review.sentiment] += 1
    
        for i in self.prior:
            self.prior[i] = self.prior[i] / self.total_reviews
        
        return self.prior
        
    def word_likelihood_calculator(self, features):
        if len(features) > 0:
            for r in self.reviews:
                r.phrase = [w for w in r.phrase if w in features]

        word_count = dict()
    
        for i in range(self.scale):  
            word_count[i] = dict()  
            for review in self.reviews:
                for word in review.phrase:
                    word_count[i][word] = 0

        for review in self.reviews:
            for word in review.phrase:
                word_count[review.sentiment][word] += 1
        
                
        likelihood_sum = dict()
        for sentiment in word_count:
            likelihood_sum[sentiment] =  sum(word_count[sentiment].values())
        
        if len(features) > 0:
            v_set = features
        else:
            v_set = []
            for s in word_count:
                v_set += word_count[s].keys()
        v_value = len(list(dict.fromkeys(v_set)))

        self.word_likelihood = word_count
        for sentiment in word_count:
            for word in word_count[sentiment]:
                self.word_likelihood[sentiment][word] = (word_count[sentiment][word] + 1) / (likelihood_sum[sentiment] + v_value)

        return self.word_likelihood


class Evaluator:
    def __init__(self, reviews, prior, likelihood, scale):
        self.reviews = reviews
        self.prior = prior
        self.likelihood = likelihood
        self.scale = scale

    def review_sentiment_calculator(self):
        self.review_score = dict()
        for i in self.prior:
            self.review_score[i] = dict()
            
        for sentiment in self.review_score:
            for review in self.reviews:
                self.review_score[sentiment][review.id] = self.prior[sentiment]
                for word in review.phrase:
                    if word in self.likelihood[sentiment]:
                        self.review_score[sentiment][review.id] = self.review_score[sentiment][review.id] * self.likelihood[sentiment][word]
        
        return self.review_score

    def review_sentiment_predictor(self):
        for review in self.reviews:
            temp = dict()
            for sentiment in self.review_score:
                temp[sentiment] = self.review_score[sentiment][review.id]
            review.sentiment = max(temp, key=temp.get) 
        return self.reviews

    def f1_calculator(self, answer_set):
        self.f1_result = dict()
        for sentiment_pred in range(self.scale):
            self.f1_result[sentiment_pred] = dict()
            for sentiment_true in range(self.scale):
                self.f1_result[sentiment_pred][sentiment_true] = 0
        
        answer_set = [a.sentiment for a in answer_set]
        predict_set = [r.sentiment for r in self.reviews]

        for i in range(len(answer_set)):
            if answer_set[i] == predict_set[i]:
                self.f1_result[answer_set[i]][answer_set[i]] += 1
            else:   
                self.f1_result[predict_set[i]][answer_set[i]] += 1
        
        return self.f1_result
    
    def score_calculator(self):
        precision_sum = dict()
        recall_sum = dict()
        for i in range(len(self.f1_result)):
            precision_sum[i] = sum(self.f1_result[i].values())
            recall_sum[i] = 0
            for j in range(len(self.f1_result)):
                recall_sum[i] += self.f1_result[j][i]

        f1_score_class = dict()
        for s in self.f1_result:
            f1_score_class[s] = 2 * self.f1_result[s][s] / (precision_sum[s] + recall_sum[s])  
  
        f1_score = sum(f1_score_class.values()) / len(f1_score_class)

        return f1_score
                 
 
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
    reviews = Review.get_reviews(training)
    reviews_preprocessed = Preprocess(reviews).preprocess_reviews()
    if number_classes == 3: 
        reviews_preprocessed = Preprocess(reviews_preprocessed).scale_3()
        classifier = Classifier(reviews_preprocessed, number_classes)
    else:
        classifier = Classifier(reviews_preprocessed, number_classes)
    prior=classifier.prior_probability()

    
    doc = []
    if features == 'features':
        for r in reviews_preprocessed:
            temp = nltk.pos_tag(r.phrase)
            for index in range(len(temp)):
                # Compare many tags result, JJ has the best performance, it means adjective or numeral, ordinal
                if temp[index][1] == 'JJ':
                    doc.append(temp[index][0])
            
    likelihood = classifier.word_likelihood_calculator(doc)

    #You need to change this in order to return your macro-F1 score for the dev set
    
    reviews_dev = Review.get_reviews(dev)
    reviews_dev_preprocessed = Preprocess(reviews_dev).preprocess_reviews()
    evaluate_dev = Evaluator(reviews_dev_preprocessed, prior, likelihood, number_classes)
    evaluate_dev.review_sentiment_calculator()
    evaluate_dev.review_sentiment_predictor()
    if number_classes == 3:
        f1_result = evaluate_dev.f1_calculator(Preprocess(Review.get_reviews(dev)).scale_3())
        if output_files:f = open('dev_predictions_3classes_acd20xh.tsv', 'w')
    else:
        f1_result = evaluate_dev.f1_calculator(Review.get_reviews(dev))  
        if output_files:f = open('dev_predictions_5classes_acd20xh.tsv', 'w')
    f1_score = evaluate_dev.score_calculator()
    if output_files:
        f.write('SentenceId\tSentiment\n')
        for r in reviews_dev_preprocessed:
            f.write(str(r.id) +'\t'+str(r.sentiment) + '\n' )
        f.close()

    if confusion_matrix:
        match number_classes:
            case 3: 
                print("%s\t%i\t%i\t%i" % ('', 0, 1, 2))
                for i in range(len(f1_result)):
                    print("%i\t%i\t%i\t%i" % (i, f1_result[i][0], f1_result[i][1], f1_result[i][2]))
            case 5:
                print("%s\t%i\t%i\t%i\t%i\t%i" % ('', 0, 1, 2, 3, 4))
                for i in range(len(f1_result)):
                    print("%i\t%i\t%i\t%i\t%i\t%i" % (i, f1_result[i][0], f1_result[i][1], f1_result[i][2], f1_result[i][3], f1_result[i][4]))

    reviews_test = Review.get_reviews(test)
    reviews_test_preprocessed = Preprocess(reviews_test).preprocess_reviews()
    evaluate_test = Evaluator(reviews_test_preprocessed, prior, likelihood, number_classes)
    evaluate_test.review_sentiment_calculator()
    evaluate_test.review_sentiment_predictor()
    if number_classes == 3:
        if output_files: f = open('test_predictions_3classes_acd20xh.tsv', 'w')
    else:
        if output_files: f = open('test_predictions_5classes_acd20xh.tsv', 'w')
    if output_files:
        f.write('SentenceId\tSentiment\n')
        for r in reviews_test_preprocessed:
            f.write(str(r.id) +'\t'+str(r.sentiment) + '\n' )
        f.close()


    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()

