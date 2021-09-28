#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is for reference and experiments, all models/code used was pulled into
notebook file in doc folder
"""
import numpy as np
import pandas as pd

#for SKlearn model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import text 
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from pyLDAvis import sklearn as sklearn_lda


from nltk.corpus import stopwords 


def find_optimal_model(transformed_text,search_params = {'n_components': [4, 5, 8, 10], 'learning_decay': [.5, .7, .9]}):

    lda_m = LDA()
    
    #Grid Search Class
    model = GridSearchCV(lda_m, param_grid=search_params)
    
    # Do the Grid Search
    model.fit(transformed_text) 
    
    best_lda_model = model.best_estimator_
    
    return model, best_lda_model
    
def print_topics(trained_lda , count_vectorizer, num_words):

    topic_list= list()
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(trained_lda.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-num_words - 1:-1]]))
        topic_list.append(" ".join([words[i]
                        for i in topic.argsort()[:-num_words - 1:-1]]))
    return topic_list


def train_lda(text_list, num_words, num_topics, stop_words_to_add):
    
    stop_words_to_add = ["thou", "thy", "thee", "doth", "thyself"]
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_to_add)
    count_vectorizer = CountVectorizer(stop_words=stop_words)
    
    # Fit and transform the processed text
    count_vec = count_vectorizer.fit_transform(text_list)
    
    # Fit LDA model on count vector 
    lda = LDA(n_components=num_topics)
    lda.fit(count_vec)
    
    #Print topics
    all_topics = print_topics(lda,count_vectorizer,num_words)
    
    return lda, all_topics
   

def create_model_vis(trained_lda_model,count_vec_fit, count_vectorizer, fp):
    
    """
    Save html visualization of topics 
    """
    prepared_vis = sklearn_lda.prepare(trained_lda_model,count_vec_fit, count_vectorizer)
    pyLDAvis.display(prepared_vis)    
    pyLDAvis.save_html(prepared_vis, fp)   
    
def get_topic_proportions(count_vec_fit, lda, orig_df):
    
    topicnames = ["Topic" + str(i) for i in range(1,lda.n_components+1)]
    topic_matr = lda.transform(count_vec_fit)
    
    df_document_topic = pd.DataFrame(np.round(topic_matr, 2), columns=topicnames)
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic["dominant_topic"] = dominant_topic+1
    
    merge_info = pd.concat([orig_df, df_document_topic], axis = 1)
    
    return merge_info 
