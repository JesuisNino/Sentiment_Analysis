# Introduction
The choice of tags clearly reflects the importance and decisiveness of adjectives for the sentiments of reviews. Compared to other words of the lexical nature, adjectives possess a stronger emotional dimension.

# Installation
The Python version is 3.10. The develop system is macOS Ventura 13.0.1 (M1).

The installation guide is showing below:
1. For reading the files, pandas is used. Install pandas by:
    * `pip3 install pandas`
2. For the preprocessing, nltk is used. Install nltk by:
    * `pip3 install nltk` / `pip3 install --user -U nltk`
3. Install the stopwords list from nltk by:
    * `python3 -m nltk.downloader stopwords`
4. This is also necessary for stopwords list working:
    * `python3 -m nltk.downloader punkt`
5. For the features selection, I used nltk.pos_tag to tag all the phrase words. Based on the tags, I used several way to select features. Thus, for installing pos_tag, please do:
    * `python3 -m nltk.downloader averaged_perceptron_tagger`


If you are Windows user or with different python versions or settings, and the above commands didn't work, try this:
1. For reading the files, pandas is used. Install pandas by:
    * `pip install pandas`
2. For the preprocessing, nltk is used. Install nltk by:
    * `pip install nltk` / `pip install --user -U nltk`
3. Install the stopwords list from nltk by:
    * `python -m nltk.downloader stopwords`
4. This is also necessary for stopwords list working:
    * `python -m nltk.downloader punkt`
5. For the features selection, I used nltk.pos_tag to tag all the phrase words. Based on the tags, I used several way to select features. Thus, for installing pos_tag, please do:
    * `python -m nltk.downloader averaged_perceptron_tagger`
6. If they are still not working. Maybe just download them on the official website. Google them is also a good idea.

# Usage
See below for a quickstart guide:

1. `python3 NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -output_files -confusion_matrix`

    where:
    * <TRAINING_FILE> <DEV_FILE> <TEST_FILE> are the paths to the training, dev and test files, respectively;
    * -classes <NUMBER_CLASSES> should be either 3 or 5, i.e. the number of classes being predicted;
    * -features is a parameter to define whether you are using your selected features or no features ('all_words' or 'features');
    * -output_files is an optional value defining whether or not the prediction files should be saved (see below – default is "files are not saved"); 
    * -confusion_matrix is an optional value defining whether confusion matrices should be shown (default is "confusion matrices are not shown").

2. Same as the installation, if the above command doesn't work, try:
    `python NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -output_files -confusion_matrix`