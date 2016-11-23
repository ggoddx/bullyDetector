
from flask import Flask, render_template, request, redirect, url_for
import sys, tweepy, Tweet_KMeans
from tweepy.parsers import JSONParser
from sklearn.externals import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import csv
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

from sklearn import svm


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('server_template.html')


@app.route('/input_from_user', methods=["POST", "GET"])
def input_from_user():
    text = request.form['stext']
    if text is "":
        return "Please enter text"
    else:
        if request.form['input'] == "username":
            return redirect(url_for('username', user_name = text))

        elif request.form['input'] == "custom":
            return redirect(url_for('custom', text = text))

        elif request.form['input'] == "search":
            return redirect(url_for('keyword',search = text ))



@app.route('/custom', methods=["POST", "GET"])
def custom():
    text = request.args.get('text')
    tweets = []
    tweet_traces = []
    tweets.append(text)

    tClf = joblib.load('traceModel.pkl')
    rClf = joblib.load('roleModel.pkl')

    traces = tClf.predict(tweets)
    roles = rClf.predict(tweets)

    kmodel = Tweet_KMeans.main()
    km = kmodel['clusterer']
    labels = kmodel['labels']
    vectorizer = kmodel['vectorizer']

    for i in range(len(tweets)):
        tr = traces[i]
        rl = roles[i]
        tw = tweets[i]
        test = [tw]
        Y = vectorizer.transform(test)
        clusters = km.predict(Y)
        lbl = labels[clusters[0]]

        tweet_traces.append((tw, tr, rl, lbl))

    return render_template("output_template.html", tweet_traces = tweet_traces)




@app.route('/username', methods=["POST", "GET"])
def username():
    tweet_traces = []

    user_name = request.args.get('user_name')

    consumer_key = 'eiDhOCZRSDY95IzZKmZVAcK20'
    consumer_secret = 'wQHu7sPvQHGfMlTDI4r3v37afB4jK3eBDHPhvXmkMjGCMsiMG0'
    access_key = '472754640-yfnNdQ3ywnkWJYNCbM7zonfgsmx3gQkVNOtAkneD'
    access_secret = 'Abfbua0ShkFtUtp2WRiaezZI7ouZEzgQmgZuAJBBAVXNO'
    #establish API

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth_handler = auth, parser = JSONParser())
    tweets = []
    api = tweepy.API(auth_handler = auth, parser = JSONParser())
    tweets = []
    userTweets = api.user_timeline(user_name)
    if not userTweets:
        return render_template("output_template.html", error="Could not find any tweets for the user!")
    for tweet in userTweets:
        tweets.append(tweet['text'])

    tClf = joblib.load('traceModel.pkl')
    rClf = joblib.load('roleModel.pkl')

    traces = tClf.predict(tweets)
    roles = rClf.predict(tweets)

    kmodel = Tweet_KMeans.main()
    km = kmodel['clusterer']
    labels = kmodel['labels']
    vectorizer = kmodel['vectorizer']

    for i in range(len(tweets)):
        tr = traces[i]
        rl = roles[i]
        tw = tweets[i]
        test = [tw]
        Y = vectorizer.transform(test)
        clusters = km.predict(Y)
        lbl = labels[clusters[0]]

        tweet_traces.append((tw, tr, rl, lbl))

    return render_template("output_template.html", tweet_traces = tweet_traces)



@app.route('/keyword', methods=["POST", "GET"])
def keyword():
    tweet_traces = []

    search = request.args.get('search')

    consumer_key = 'eiDhOCZRSDY95IzZKmZVAcK20'
    consumer_secret = 'wQHu7sPvQHGfMlTDI4r3v37afB4jK3eBDHPhvXmkMjGCMsiMG0'
    access_key = '472754640-yfnNdQ3ywnkWJYNCbM7zonfgsmx3gQkVNOtAkneD'
    access_secret = 'Abfbua0ShkFtUtp2WRiaezZI7ouZEzgQmgZuAJBBAVXNO'
    #establish API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth_handler = auth, parser = JSONParser())
    tweets = []
    searchTweets = api.search(search, 'en')

    for tweet in searchTweets['statuses']:
        if tweet['text'][0:2] != 'RT':
            tweets.append(tweet['text'])

    if not tweets:
        return render_template("output_template.html", error="Could not find any tweets for the search term!")

    tClf = joblib.load('traceModel.pkl')
    rClf = joblib.load('roleModel.pkl')

    traces = tClf.predict(tweets)
    roles = rClf.predict(tweets)

    kmodel = Tweet_KMeans.main()
    km = kmodel['clusterer']
    labels = kmodel['labels']
    vectorizer = kmodel['vectorizer']

    for i in range(len(tweets)):
        tr = traces[i]
        rl = roles[i]
        tw = tweets[i]
        test = [tw]
        Y = vectorizer.transform(test)
        clusters = km.predict(Y)
        lbl = labels[clusters[0]]
        tweet_traces.append((tw, tr, rl, lbl))

    return render_template("output_template.html", tweet_traces = tweet_traces)


if __name__ == '__main__':

    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''


    lemmatiser = WordNetLemmatizer()

    def tokenizer(text):

        lemmatized_words = []
        tokens = word_tokenize(text)
        tokens_pos = pos_tag(tokens)
        count = 0
        for token in tokens:
            pos = tokens_pos[count]
            pos = get_wordnet_pos(pos[1])
            if pos != '':
                lemma = lemmatiser.lemmatize(token, pos)
            else:
                lemma = lemmatiser.lemmatize(token)
            lemma = lemma + "_" + tokens_pos[count][1]
            lemmatized_words.append(lemma)
            count+=1
        return lemmatized_words



    with open("Train_Bully2.csv","r") as f:
        train_text = f.readlines()
    f.close()


    with open("Test_Bully1.csv","r") as f:
        test_text = f.readlines()

    f.close()

    train_data = []
    train_target = np.ndarray(shape = len(train_text), dtype ='int64')
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    labels = []

    with open("Train_Bully2.csv","r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tweet = row[0]
            tweet = tweet.lower()
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)    # Try replacing it with HTTPLINK
            tweet = re.sub('@[^\s]+','USER_NAME',tweet)
            tweet = re.sub('[\s]+', ' ', tweet)
            tweet = pattern.sub(r"\1\1",tweet)
            tweet = re.sub("[.,?]"," ",tweet)
            train_data.append(tweet)
            labels.append(int(row[1].strip('"')))

    f.close()

    train_target = np.asarray(labels, dtype='int64')
    linear_pipe = Pipeline([('vect', CountVectorizer(encoding="latin-1", ngram_range=(1,2), analyzer='word', tokenizer=tokenizer,
                                                 )),('tfidf', TfidfTransformer(use_idf="True")), ('clf', svm.LinearSVC()),])



    test_data = []
    test_target = []
    labels = []

    with open("Test_Bully1.csv","r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tweet = row[0]

            tweet = tweet.lower()
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
            tweet = re.sub('@[^\s]+','USER_NAME',tweet)
            tweet = re.sub('[\s]+', ' ', tweet)

            tweet = pattern.sub(r"\1\1",tweet)

            tweet = re.sub("[.,?]"," ",tweet)
            test_data.append(tweet)
            labels.append(int(row[1].strip('"')))

    test_target = np.asarray(labels, dtype='int64')


    gs_svm = linear_pipe.fit(train_data, train_target)

    filename = 'traceModel.pkl'
    _ = joblib.dump(gs_svm, filename, compress=9)

    with open("Train_Author.csv","r") as f:
        train_text = f.readlines()
    f.close()

    with open("Test_Author.csv","r") as f:
        test_text = f.readlines()
    f.close()

    train_data = []
    train_target = np.ndarray(shape = len(train_text), dtype ='int64')
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    labels = []

    #Preprocessing on the tweets
    with open("Train_Author.csv","r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tweet = row[0]
            tweet = tweet.lower()
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)    # Try replacing it with HTTPLINK
            tweet = re.sub('@[^\s]+','USER_NAME',tweet)
            tweet = re.sub('[\s]+', ' ', tweet)
            tweet = tweet.strip('\'"')
            tweet = pattern.sub(r"\1\1",tweet)
            tweet = re.sub("[.,?]"," ",tweet)
            train_data.append(tweet)
            labels.append(int(row[1].strip('"')))

    f.close()

    train_target = np.asarray(labels, dtype='int64')
    linear_pipe = Pipeline([('vect', CountVectorizer(encoding="latin-1", ngram_range=(1,2), analyzer='word',)),('tfidf', TfidfTransformer(use_idf="True")), ('clf',svm.LinearSVC())])

    test_data = []
    test_target = []
    labels = []

    with open("Test_Author.csv","r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tweet = row[0]
            tweet = tweet.lower()
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
            tweet = re.sub('@[^\s]+','USER_NAME',tweet)
            tweet = re.sub('[\s]+', ' ', tweet)
            tweet = tweet.strip('\'",.')
            tweet = pattern.sub(r"\1\1",tweet)
            tweet = re.sub("[.,?]"," ",tweet)
            test_data.append(tweet)
            labels.append(int(row[1].strip('"')))

    test_target = np.asarray(labels, dtype='int64')

    gs_svm = linear_pipe.fit(train_data, train_target)
    filename = 'roleModel.pkl'
    _ = joblib.dump(gs_svm, filename, compress=9)

    app.run()


























