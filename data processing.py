from typing import List

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Counter
from wordcloud import WordCloud
from Naive_Bayes_Classifier import NBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
# with open('1kEmails.csv') as f:
#     df = csv.reader(f)
#     row: list[str]
#     for row in df:
#         print(row)

df = pd.read_csv('1kEmails.csv')
df.columns = ['Label', 'Mail']

print(df.shape)
df.head()

df.drop_duplicates(inplace = True)
print(df.shape)

randomized = df.sample(frac=1, random_state = 1)
train_size = round(len(randomized) * 0.8)
train = randomized[:train_size].reset_index(drop=True)
test = randomized[train_size:].reset_index(drop=True)
print(train.shape)
print(test.shape)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def process_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    text = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text])
    text = text.replace('’ ', '')

    text = text.split()

    return text


print(process_text("Dear Voucher Holder, To claim this weeks offer, at you PC please go to http://www.e-tlp.co.uk/expressoffer Ts&Cs apply. To stop texts, txt STOP to 80062"))

bag_word = []
testmails= test['Mail']
test_labels = test['Label']
testmails2 = []
trainmails = train['Mail']
train_labels = train['Label']
trainmails2= []

testmails = testmails.apply(process_text)
trainmails = trainmails.apply(process_text)

text = ""

for message in trainmails:
    text = text + " ".join(message) + " "
text1 = text.split()
text1 = set(text1)
print(len(text1))
wordcloud = WordCloud(width=1200,height=800,collocations = False,background_color='white').generate(text)

plt.figure(figsize=(12,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')

text = ""

for message in train[train.Label == 0].Mail:
    words = message.split()
    text = text + " ".join(words) + " "
text1 = text.split()
text1 = set(text1)
print(len(text1))
wordcloud = WordCloud(width=1200,height=800,collocations = False,background_color='white').generate(text)

plt.figure(figsize=(12,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')

text = ""

for message in train[train.Label == 1].Mail:
    words = message.split()
    text = text + " ".join(words) + " "

text1 = text.split()
text1 = set(text1)
print(len(text1))
wordcloud = WordCloud(width=1200,height=800,collocations = False,background_color='white').generate(text)

plt.figure(figsize=(12,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
