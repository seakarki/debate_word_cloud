# Reading the content (it is divided in paragraphs)
# importing the necessary packages
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import os
import re

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import urllib
import requests
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#sentence_words = nltk.word_tokenize(str(trump_text))

stop_words = ['let','will','go','going','let','want','Mr','President','sir','question','Vice',
             'right','say','okay','said','got','look','saying','two','minute','first','man','ask','asking',
             'asked','one','let','three','know','done','able','much','now','think','much','many','lot','good',
             'talk','talking','see','oh',"n't"] + list(STOPWORDS)

mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/W/J/8/6/8/Y/dude-hi.png', stream=True).raw))

# class debate:
#     # init method or constructor
#     def __init__(self,url,class_):
#         self.url = url
#         self.class_ = class_
    
    # method or function

def has_speaker(sentence):
    try:
        name = sentence.split(' ')[0]
        return name[:-1] if name.isupper() and ':' in name else None
    except:
        return None

def scrap_debate(url,class_):
    article = requests.get(url,)
    article_content = article.content
    soup_article = BeautifulSoup(article_content, 'html5lib')
    body = soup_article.find_all('div', class_=class_)
    x = body[0].find_all('p')
    data = []
    for p in np.arange(0, len(x)):
        paragraph = x[p].get_text() 
#         xt= nltk.word_tokenize(str(paragraph))
#         xt = [wordnet_lemmatizer.lemmatize(word) for word in xt]
#         xt = ' '.join(xt)
        #print(xt)
        #print('\n',paragraph)
        data.append(paragraph)
        speakers = defaultdict(list)
        current_speaker = None
        for sentence in data:
            speaker = has_speaker(sentence)    
            if speaker is not None:
                current_speaker = speaker
                sentence = sentence[len(speaker) + 2:]
            speakers[current_speaker].append(sentence)
    cc = str(debate_text_process(speakers))
    dd = generate_wordcloud(cc, mask)
    return dd
    #return speakers

def debate_text_process(text):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(str(text))
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    from spacy.lang.en.stop_words import STOP_WORDS
    stop_words = set(stopwords.words('english'))

    STOP_WORDS.update(stop_words)
    STOP_WORDS.update({'nt','okay','ha','thank','wa','got','oh','said','going','want','let','know'})
    words = [w for w in words if not w in STOP_WORDS]
    #print(len(STOP_WORDS))

    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    return words


# This function takes in your text and your mask and generates a wordcloud. 
def generate_wordcloud(words, mask):
    word_cloud = WordCloud(width = 40, height = 512, background_color='white',collocations=False, stopwords=stop_words, mask=mask).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'black', edgecolor='black')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=2)
    plt.show()
    

# def main(url,class_):
#     scrap_debate_text = scrap_debate(url,class_)
#     scrap_debate_tokens = str(debate_text_process(scrap_debate_text))
#     wordcloud_pic = generate_wordcloud(scrap_debate_tokens, mask)
#     return wordcloud_pic
    
import argparse
parser = argparse.ArgumentParser(description='Create a Wordcloud from debate text')
parser.add_argument('url', type=str, metavar=' ', help='the url of debate')
parser.add_argument('class_',  type=str,metavar=' ',  help=' class from the web scraping app')
args = parser.parse_args()

# scrap_debate(workspace=args.url, schema=args.class_)
# scrap_debate(url,class_)



if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser(description='Create a Wordcloud from debate text')
    # parser.add_argument('url', type=str, metavar=' ', help='the url of debate')
    # parser.add_argument('class_',  type=str,metavar=' ',  help=' class from the web scraping app')
    # args = parser.parse_args()
    scrap_debate(args.url, args.class_)




    
