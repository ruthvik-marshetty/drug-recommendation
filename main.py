

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
import string
from nltk.corpus import stopwords
import nltk
import regex as re
#nltk.download('stopwords')
#nltk.download('vader_lexicon')
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler
import joblib
from scipy.sparse import hstack
import spacy

import flask
app = Flask(__name__)

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()
stop_words.remove('no')
entity_keys = ['TIME','WORK_OF_ART']
ner_lst = nlp.pipe_labels['ner']

#loading the pretrained models
clf2 = joblib.load('Bow_model.pkl')
clf3 = joblib.load('tfidf_model.pkl')
clf4 = joblib.load('ngram_bow_model.pkl')
clf5 = joblib.load('ngram_tfidf_model.pkl')
clf6 = joblib.load('W2V Model.pkl')


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(text_data):
    
    text_data = decontracted(text_data)
    
    text_data = text_data.replace('\n',' ')
    text_data = text_data.replace('\r',' ')
    text_data = text_data.replace('\t',' ')
    text_data = text_data.replace('-',' ')
    text_data = text_data.replace("/",' ')
    text_data = text_data.replace(">",' ')
    text_data = text_data.replace('"',' ')
    text_data = text_data.replace('?',' ')
    return text_data


def nlp_preprocessing(review):
    '''This functional block preprocess the text data by removing digits, extra spaces, stop words 
    and converting words to lower case and stemming words'''
    
    
    if type(review) is not int:
        string = ""
        review = preprocess_text(review)
        review = re.sub('[^a-zA-Z]', ' ', review)
        
        review = re.sub('\s+',' ', review)
        
        review = review.lower()
        
        for word in review.split():
        
            if not word in stop_words:
                word = stemmer.stem(word)
                string += word + " "
        
        return string 


def get_sentiment_score(review,cleaned_review):

    rev_score = sid.polarity_scores(review)['compound']
    clean_rev_score = sid.polarity_scores(review)['compound']
    
    return rev_score,clean_rev_score


def get_extracted_features(review,cleaned_review):
    
    #reference from quora question pair case study

    #Word count in each review
    word_count =  len(str(cleaned_review).split())

    #Unique word count 
    unique_word_count = len(set(str(cleaned_review).split()))

    #character count
    char_length = len(str(cleaned_review))

    #punctuation count
    count_punctuations = len([c for c in str(review) if c in string.punctuation])


    #Number of stopwords
    stopword_count = len([w for w in str(review).lower().split() if w in stop_words])

    #Average length of the words
    mean_word_len = np.mean([len(w) for w in str(cleaned_review).split()])
    
    return word_count,unique_word_count,char_length,count_punctuations,stopword_count,mean_word_len
    



def ner(review):

    sent = review
    doc=nlp(sent)
    dic = {}.fromkeys(ner_lst,0)
    for word in doc.ents:
        dic[word.label_]+=1
        
    return dic

def get_topic_modelling_features(review):
    lst =[]
    lst.append(review.split())
    id2word = gensim.corpora.Dictionary(lst) ## map words to an id
    dic_corpus = [id2word.doc2bow(word) for word in lst] ## create dictionary word:freq
    ## train LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, num_topics=20, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    top_topics = (lda_model.get_document_topics(dic_corpus[0], minimum_probability=0.0))
    topic_vec = [top_topics[i][1] for i in range(20)]

    
    return topic_vec



def normalize_num_features(features):
    normalizer = Normalizer()
    num_feat = normalizer.fit_transform(features)
    
    return num_feat

def create_w2v(review,model):
    '''This function creates the w2v embeddings for the cleaned reviews passed'''
    w2v_vector =[]
    vector = np.zeros(300)
    for word in review.split():
        if word in model.wv.key_to_index:
            vector += model.wv[word]
    
    w2v_vector.append(vector)
    w2v_vector = np.array(w2v_vector)
    return w2v_vector




@app.route('/')
def home():
    return flask.render_template('home.html')


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/recommendation')
def recommendation():
    return flask.render_template('recommendation.html')


@app.route('/recommend', methods=['POST'])
   
    
def recommend():
    '''this function takes the given condition as input and returns the top drugs based on highest rec scores'''
    to_predict_list = request.form.to_dict()
    condition = to_predict_list['condition']
    data = pd.read_csv('validation_data.csv')
    select = data[data['condition']==condition]
    drug_count = select['drugName'].nunique()
    select['rec_score'] = select['rec_score']/drug_count
    group_drug = select.groupby(['drugName']).agg({'rec_score':['sum']})
    group_drug = group_drug[('rec_score', 'sum')].sort_values(ascending=False)
    drug_score = dict(group_drug)
    drugs =[]
    if len(drug_score)>5:
        for i in list(drug_score.keys())[0:5]:
            drugs.append(i)
        return flask.render_template('drugs.html',table=drugs) 
            
    else:
        for i in drug_score.keys():
            drugs.append(i)
            
        return flask.render_template('drugs.html',table=drugs)
    

@app.route('/predict', methods=['POST'])
def predict():
    # reading the input data
    to_predict_list = request.form.to_dict()
    review = to_predict_list['review_text']
    condition = to_predict_list['condition']
    year = int(to_predict_list['year'])
    usefulcount = np.array(int(to_predict_list['usefulcount'])).reshape(1,-1)
    
    #preprocessing the review_text
    cleaned_review = nlp_preprocessing(review)
    scores = np.array(get_sentiment_score(review,cleaned_review)).reshape(1,-1)
    extracted_features  = np.array(get_extracted_features(review,cleaned_review)).reshape(1,-1)
    entities = np.array([ner(cleaned_review).get(key) for key in entity_keys]).reshape(1,-1)
    topics = get_topic_modelling_features(cleaned_review)
    del(topics[9])
    del(topics[14])
    topics = np.array(topics).reshape(1,-1)
    
    #normalizing and concatenating numerical features 
    num_features = np.concatenate((usefulcount,extracted_features,entities,topics),axis=1)
    norm_features = normalize_num_features(num_features)
    num_features = np.concatenate((norm_features,scores),axis=1)
    
    #encoding categorical features 
    label_con = joblib.load('condition_encoder.pkl')
    condition = np.array(label_con.transform([condition])).reshape(1,-1)
    label_year = joblib.load('year_encoder.pkl')
    year = np.array(label_year.transform([year])).reshape(1,-1)
    
    #loading the predefined vectorizers
    vectorizer_bow_1  = joblib.load('vectorizer_bow.pkl')
    vectorizer_tfidf_1 = joblib.load('vectorizer_tfidf.pkl')
    vectorizer_bow_n = joblib.load('ngram_vec_bow.pkl')
    vectoizer_tfidf_n = joblib.load('ngram_vec_tfidf.pkl')
    vectorizer_w2v = joblib.load('word2vec.bin')
    
    #transforming the cleaned review
    vec_bow_1 = vectorizer_bow_1.transform([cleaned_review])
    vec_tfidf_1 = vectorizer_tfidf_1.transform([cleaned_review])
    vec_bow_n = vectorizer_bow_n.transform([cleaned_review])
    vec_tfidf_n = vectoizer_tfidf_n.transform([cleaned_review])
    vec_w2v = create_w2v(cleaned_review, vectorizer_w2v)
    
    # concatenating all the features     
    vector2 = hstack((num_features,condition,year,vec_bow_1)).tocsr()
    vector3 = hstack((num_features,condition,year,vec_tfidf_1)).tocsr()
    vector4 = hstack((num_features,condition,year,vec_bow_n)).tocsr()
    vector5 = hstack((num_features,condition,year,vec_tfidf_n)).tocsr()
    vector6 = np.concatenate((num_features,condition,year,vec_w2v),axis=1)
    

    
 
    #predicting the output for given query point
    pred =[]
    pred.append(clf2.predict(vector2)[0])
    pred.append(clf3.predict(vector3)[0])
    pred.append(clf4.predict(vector4)[0])
    pred.append(clf5.predict(vector5)[0])
    pred.append(clf6.predict(vector6)[0])
    
    
    if sum(pred)>=3:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)






