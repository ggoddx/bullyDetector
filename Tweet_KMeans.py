# Credit: Peter Prettenhofer <peter.prettenhofer@gmail.com> Lars Buitinck <L.J.Buitinck@uva.nl>

from __future__ import print_function
#from stop_words import get_stop_words
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import csv, re
from nltk.corpus import stopwords

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import codecs

def main():

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa", dest="n_components", type="int", help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch", action="store_false", dest="minibatch", default=True, help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf", action="store_false", dest="use_idf", default=True, help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing", action="store_true", default=False, help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000, help="Maximum number of features (dimensions) to extract from text.")
    op.add_option("--verbose", action="store_true", dest="verbose", default=False, help="Print progress reports inside k-means algorithm.")

    # print(__doc__)
    # op.print_help()

    (opts, args) = op.parse_args()
    # if len(args) > 0:
    #     op.error("this script takes no arguments.")
    #     sys.exit(1)


    ###############################################################################
    # Load some categories from the training set
    #categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space',]
    # Uncomment the following to do the analysis on all the categories
    #categories = None



#    if (len(sys.argv) > 1):
#        fname= sys.argv[1]
#    else:
        #        print ('No file specified. Using test data which has been annotated as one with bullying trace')
#        fname = "tweet_topic_clustering.csv"
    fname = "manual_topics.csv"
    texts = []
    clustNames = []
    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            texts.append(row[0])
            clustNames.append(row[1])

    #print (texts)

    listoftweets = []
    for row in texts:
        try:
            line = row.encode('utf-8')
            listoftweets.append(line)
        except:
            continue

#    en_stop = get_stop_words('en')
    en_stop= stopwords.words('english')
    listofprocessedtweets = []

    for row in listoftweets:
        row = row.lower()

        row = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',row)
        row = re.sub('@[^\s]+','',row)
        row = re.sub('[\s]+', ' ', row) # remove multiple consecutive blank space
        row = re.sub(r'#([^\s]+)', "", row)
        row = re.sub('"',"", row)
        row = row.strip()
        # print row
        # row = row.strip('\'"')
        row = re.sub("[^a-zA-Z']"," ",row)
        # print row
        row = re.sub(' +',' ',row)
        row = re.sub(r'(.)\1+', r'\1\1', row)
        row = re.sub("[.,!?]"," ", row)
        # row = re.sub(r'\W*\b\w{1}\b', "", row)
        row = re.sub('bull[^\s]+','',row)
        # print row
        listofprocessedtweets.append(row)

        #print (listofprocessedtweets[0:3])

    text = []
    for tweet in listofprocessedtweets:
        listoftweetwords=tweet.split()
        listofgoodwords = [word for word in listoftweetwords if not word in en_stop]
        text.append(" ".join(listofgoodwords))

    true_k = 7

    ###############################################################################

	# Perform an IDF normalization on the output of HashingVectorizer
    hasher = HashingVectorizer(n_features=opts.n_features, stop_words='english', non_negative=True, norm=None, binary=False)
    #print ("Vectorizer 1")
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    
    X = vectorizer.fit_transform(text)

    ###############################################################################
    # Do the actual clustering

    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=10, n_init=1, verbose=opts.verbose)
    km.fit(X)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    ###############################################################################

    labels = []
    for i in range(true_k):
        labels.append("test")
    for i in range(true_k):
        Y = vectorizer.transform([texts[i].encode('utf-8')])
        cluster = km.predict(Y)
        # print (cluster[0], " ", clustNames[cluster[0]])
        labels[cluster[0]] = clustNames[i]
    
    return {'clusterer': km, 'labels': labels, 'vectorizer': vectorizer}


    # if (len(sys.argv) > 1):
    #     test= sys.argv[1]
    #     print (test)
    # else:
    #     print (u'No tweet specified. Using test tweet - herd that you got  when you was years old and i feel bad are you ok')
    #     test = [u'herd that you got  when you was years old and i feel bad are you ok']


    '''test = [u'herd that you got  when you was years old and i feel bad are you ok']
    test = [u"default tweet"]
    while test != [u"quit"] and test != [u"Quit"] and test != [u"QUIT"]:
        test = [raw_input("Enter tweet: ").encode('utf-8')]
        Y = vectorizer.transform(test)
        clusters = km.predict(Y)
        i = clusters[0]
        print("Cluster %d, " % i, end='')
        print("Topic label " + labels[i])'''
    # for ind in order_centroids[i, :15]:
    # print(' %s' % terms[ind], end='')
    # print()

if __name__ == '__main__':
    main()
