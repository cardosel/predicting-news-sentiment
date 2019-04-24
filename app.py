# Usage: python2 app.py "homework.json"
import pandas as pd
import json
import subprocess
import numpy as np
import sys
src = sys.argv[1]
output = sys.argv[2]

lines = []
for line in open(src, 'r'):
    lines.append(json.loads(line))

data = pd.DataFrame(lines)

# drop nulls from the data
data = data.dropna()

# get rid of any string values

data['emotion_0'] = data['emotion_0'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_1'] = data['emotion_1'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_2'] = data['emotion_2'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_3'] = data['emotion_3'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_4'] = data['emotion_4'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_5'] = data['emotion_5'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_6'] = data['emotion_6'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_7'] = data['emotion_7'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_8'] = data['emotion_8'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 
data['emotion_9'] = data['emotion_9'].apply(lambda x: "0.0" if 'cat' in str(x) else("0.0" if 'fnord' in str(x) else x)) 

data.iloc[:,0:10] = data.iloc[:,0:10].astype("float")

# boilerplate code for forming bag of words vectors from headlines
import nltk
import numpy
import re
from nltk.corpus import stopwords
set(stopwords.words('english'))

BoW = []
def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text    
    
def generate_bow(allsentences):    
    vocab = tokenize(allsentences)
    #print("Word List for Document \n{0} \n".format(vocab));

    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab):
                if word == w: 
                    bag_vector[i] += 1
                    
        #print("{0} \n{1}\n".format(sentence,numpy.array(bag_vector)))
        BoW.append(numpy.array(bag_vector))
        
    
allsentences = data.headline.str.encode('utf-8').sample(1000)

generate_bow(allsentences)

vocab = tokenize(allsentences)
bow_df = pd.DataFrame(BoW)

bow_w_headlines = pd.concat([allsentences.reset_index(), bow_df], axis=1)
bow_w_headlines2 = bow_w_headlines.merge(data, on='headline', how='left')
bow_w_headlines2 = bow_w_headlines2.dropna()


bowshape = bow_w_headlines2.shape[1]

bow_w_headlines2.iloc[:,2:bowshape-2].head()
bow_w_headlines2.iloc[:,2:bowshape-2] = bow_w_headlines2.iloc[:,2:bowshape-2].astype("float")

bow_w_headlines_ft_matrix = bow_w_headlines2.groupby('headline').mean().reset_index()



X = bow_w_headlines_ft_matrix.iloc[:,2:bowshape-2-10]
y = bow_w_headlines_ft_matrix.iloc[:,bowshape-2-10:bowshape-2]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


rf = RandomForestRegressor(n_estimators=100, random_state=10)

preds = {}
for model_name, model in zip(['RandomForestRegressor'], [rf]):
    model.fit(X_train, y_train)
    preds[model_name] = model.predict(X_test)

# Evaluating the Algorithm
from sklearn import metrics
for k in preds:
    print("{} performance:".format(k))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, preds[k]))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, preds[k]))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, preds[k])))
    print " "

sample = np.random.randint(50000,93758)
str(data.headline.iloc[sample:sample+1,].reset_index().headline[0]).decode('utf-8')
unseen_doc = []
sentence = str(data.headline.iloc[sample:sample+1,].reset_index().headline[0])
words = word_extraction(sentence)
bag_vector_sample = numpy.zeros(len(vocab))
for w in words:
    for i,word in enumerate(vocab):
                if word == w: 
                    bag_vector_sample[i] += 1
                    
print("{0} \n{1}\n".format(sentence,numpy.array(bag_vector_sample)))
unseen_doc.append(numpy.array(bag_vector_sample))
unseen_doc = pd.DataFrame(unseen_doc)

print "Random Forest Regressor Predictions: "
dict2 = []
for i in range(len(y.columns.tolist())):
	print " "
	dict2.append(str(y.columns.tolist()[i])+": "+str(rfc.predict(unseen_doc).tolist()[0][i]))

dict2 = pd.DataFrame(dict2)
dict2.to_csv(output, index=False)





