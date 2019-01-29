import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import Word2Vec
import gensim
import nltk
from random import shuffle
import zipfile
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', -1)
import theano
import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical

train=pd.read_csv('train.csv')

tag=sorted(train.Tag.unique())
tagmapping=dict(zip(tag,np.arange(0,len(tag))))
train['Tag']=train['Tag'].map(tagmapping).astype(int)


def preprocess(url,title):
    
    pattern=r'//.*'
    urls=[]
    for i in url:
        b=re.findall(pattern,i)
        c=b[0][2:]
        d=re.split('\.|/|-|_',c)
        e=''
        for i in d:
            e+=str(i).lower()
            e+=' '
        e = ''.join([i for i in e if not i.isdigit()])
        f=''
        for i in e:
            if i.isalpha() or i==' ':
                f+=i
            else:
                f+=' '
        urls.append(f)
        
    titles=[]
    for e in title:    
        e = ''.join([i for i in e if not i.isdigit()])
        f=''
        for i in e:
            if i.isalpha() or i==' ':
                #print(i)
                f+=i
            else:
                f+=' '
        f = re.sub(' +',' ',f) # replace series of spaces with single space
        titles.append(f)
    
    data=[]
    for i in range(len(title)):
        s=titles[i]+' '+urls[i]
        data.append(str(s))
                    
    return(data)


def preprocess_title():
    pattern=r'<title>.*</title>'

    zf = zipfile.ZipFile('train.zip') 
    df = pd.read_csv(zf.open('html_data.csv'),chunksize=1)

    title=[]
    chunksize = 1
    for chunk in pd.read_csv(zf.open('html_data.csv'), chunksize=chunksize):
        a=chunk.Html
        idd=chunk.Webpage_id
        c=re.findall(pattern,a.iloc[0])
        if len(c)!=0:
            title.append([idd.iloc[0],c[0][7:-8]])
        else:
            title.append([idd.iloc[0],' '])
    return(title)


title=preprocess_title()


title_data=pd.DataFrame(title)
title_data.columns=['Webpage_id','Html']

new_data=pd.merge(train,title_data,on=['Webpage_id'],how='left')
new_data.to_csv('processed_train_data.csv',index=False)

data=pd.read_csv('processed_train_data.csv',lineterminator='\n')     #saved file to save time

tag=sorted(data.Tag.unique())
tagmapping=dict(zip(tag,np.arange(0,len(tag))))
data['Tag']=data['Tag'].map(tagmapping).astype(int)


def train_val_split(train):
    grp=train.groupby('Tag')
    
    test_set=[]
    train_set=[]
    test_tag_set=[]
    train_tag_set=[]
    count=0
    for unique_tags in train.Tag.unique():
        tag=grp.get_group(unique_tags)
        s=set(tag.Domain)
        count+=len(s)
        a=list(s)
        #shuffle(a)
        for i in range(len(a)):
            if (i+1)%3==0:
                test_set.append(a[i])
                test_tag_set.append(unique_tags)
            else:
                train_set.append(a[i])
                train_tag_set.append(unique_tags)
    
    
    train_domain=pd.DataFrame(train_set)
    train_domain.columns=['Domain']
    train_domain['Tag']=train_tag_set
    train_data=pd.merge(train_domain,train,on=['Domain','Tag'],how='left')
    
    val_domain=pd.DataFrame(test_set)
    val_domain.columns=['Domain']
    val_domain['Tag']=test_tag_set
    val_data=pd.merge(val_domain,train,on=['Domain','Tag'],how='left')
    
    return(train_data,val_data)
    

train_data,val_data = train_val_split(data)

train_urls=preprocess(train_data.Url,train_data['Html\r'])
train_target=train_data.Tag.values

val_urls = preprocess(val_data.Url,val_data['Html\r'])
val_target = val_data.Tag.values

def more_prep(urls):
    url=[]
    stop_words = set(['','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]) 

    for i in urls:
        c = re.sub(r"[^a-z A-Z]+", "", i)
        c = re.sub('www','',c)
        c = c.lower()
        words = c.split(' ')
        filtered_sentence = [w for w in words if not w in stop_words]   
        url.append(filtered_sentence)
        
    return(url)

train_url=more_prep(train_urls)
val_url=more_prep(val_urls)

def unique(data):
    result=[]
    for sentence in data:
        seen = set()
        result.append([x for x in sentence if not (x in seen or seen.add(x))])
    return(result)

train_url=unique(train_url)
val_url=unique(val_url)

def ohe(target):
    n=np.arange(0,9).reshape((-1,1))
    #print(n)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(n)
    y_enc=enc.transform(np.array(target).reshape((target.shape[0],1))).toarray()
    return(y_enc)


y_train_ohe=to_categorical(train_target, num_classes=9)
y_val_ohe=to_categorical(val_target, num_classes=9)


with open('urls.pkl', 'rb') as f:
    urls = pickle.load(f)

url=more_prep(urls)

mod = Word2Vec(url)    #Training Word2Vec model on all data')
print(len(mod.wv.vocab))

#LSTM

def sentences_to_indices(text , mod, max_len):
    m = len(text)                                 
    text_indices = np.zeros((m, max_len))
    
    for i in range(m):                      
        j = 0
        for w in text[i]:
            if j==max_len:
                break
            if w not in mod.wv.vocab:
                continue
            text_indices[i, j] = mod.wv.vocab[w].index  # Set the (i,j)th entry of X_indices to the index of the correct word.
            j = j + 1
            
    return text_indices


def pretrained_embedding_layer(mod):
    vocab_len = len(mod.wv.vocab) + 1                 
    
    emb_dim = mod["hi"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    index=0
    for word in mod.wv.vocab:
        emb_matrix[index, :] = mod[word]
        index+=1
        
    embedding_layer = Embedding(vocab_len, emb_dim)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def webpage_model(input_shape,mod):
    sentence_indices = Input(shape=input_shape)
    embedding_layer =  pretrained_embedding_layer(mod)
    embeddings = embedding_layer(sentence_indices)   
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    #X = Flatten()(X)
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)  
    X = Dense(9, activation='softmax')(X)
    X =  Activation('softmax')(X)
    model = Model(sentence_indices, X)
    return model


max_len=10
model = webpage_model((max_len,),mod)
print(model.summary())

from keras import optimizers
adm = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adm , metrics=['accuracy'])

x_train_indices = sentences_to_indices(train_url, mod,max_len)
x_val_indices = sentences_to_indices(val_url, mod,max_len)


model.fit(x_train_indices, y_train_ohe, epochs = 50, batch_size = 32, shuffle=True)
filename = 'model1.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

model.evaluate(x_val_indices,y_val_ohe)

