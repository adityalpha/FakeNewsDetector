from psaw import PushshiftAPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""%config InlineBackend.figure_format = 'retina'
%matplotlib inline"""
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
def clean_data(dataframe):

    # Drop duplicate rows
    dataframe.drop_duplicates(subset='title', inplace=True)
    
    # Remove punctation
    dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

    # Remove numbers 
    dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

    # Make sure any double-spaces are single 
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')

    # Transform all text to lowercase
    dataframe['title'] = dataframe['title'].str.lower()
    
    print("New shape:", dataframe.shape)
    #return dataframe.head()




df_onion = pd.read_csv('./data/the_onion.csv')
df_not_onion = pd.read_csv('./data/not_onion.csv')

clean_data(df_onion)
clean_data(df_not_onion)

df = pd.concat([df_onion[['subreddit', 'title']], df_not_onion[['subreddit', 'title']]], axis=0)
df = df.reset_index(drop=True)

df["subreddit"] = df["subreddit"].map({"nottheonion": 0, "TheOnion": 1})

print(df.shape)
#doing a unigram count_vectorizer to count individual words.
mask_on = df['subreddit'] == 1
df_onion_titles = df[mask_on]['title']

cv1 = CountVectorizer(stop_words = 'english')

onion_cvec = cv1.fit_transform(df_onion_titles)

# Convert onion_cvec into a DataFrame
onion_cvec_df_u = pd.DataFrame(onion_cvec.toarray(),columns=cv1.get_feature_names())

# Set up variables to contain top 5 most used words in Onion
onion_wc = onion_cvec_df_u.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(15)

# Set variables to show NotTheOnion Titles
mask_no = df['subreddit'] == 0
df_not_onion_titles = df[mask_no]['title']

cv2 = CountVectorizer(stop_words = 'english')

not_onion_cvec = cv2.fit_transform(df_not_onion_titles)

not_onion_cvec_df_u = pd.DataFrame(not_onion_cvec.toarray(),columns=cv2.get_feature_names())

nonion_wc = not_onion_cvec_df_u.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(15)

# Create list of unique words in top five
not_onion_5_set = set(nonion_top_5.index)
onion_5_set = set(onion_top_5.index)

common_unigrams = onion_5_set.intersection(not_onion_5_set)

#Bigram
mask = df['subreddit'] == 1
df_onion_titles = df[mask]['title']

cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))

onion_cvec = cv.fit_transform(df_onion_titles)

onion_cvec_df_b = pd.DataFrame(onion_cvec.toarray(),columns=cv.get_feature_names())

mask = df['subreddit'] == 0
df_not_onion_titles = df[mask]['title']

cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))

not_onion_cvec = cv.fit_transform(df_not_onion_titles)
#print(type(not_onion_cvec))
not_onion_cvec = not_onion_cvec.astype(dtype=np.int32, casting='unsafe', copy=True)
not_onion_cvec_df_b = pd.DataFrame(not_onion_cvec.toarray() , columns=cv.get_feature_names())

onion_wc = onion_cvec_df_b.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(15)

nonion_wc = not_onion_cvec_df_b.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(15)

not_onion_5_list = set(nonion_top_5.index)
onion_5_list = set(onion_top_5.index)

common_bigrams = onion_5_list.intersection(not_onion_5_list)

custom = stop_words.ENGLISH_STOP_WORDS
custom = list(custom)
common_unigrams = list(common_unigrams)
common_bigrams = list(common_bigrams)

for i in common_unigrams:
    custom.append(i)
    
for i in common_bigrams:
    split_words = i.split(" ")
    for word in split_words:
        custom.append(word)

print(df['subreddit'].value_counts(normalize=True))

from sklearn.model_selection import train_test_split
X = df['title']
y = df['subreddit']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,stratify=y)

from sklearn.naive_bayes import MultinomialNB
#Instantiate the classifier and vectorizer
nb = MultinomialNB(alpha = 0.36)
cvec = CountVectorizer(ngram_range= (1, 3))

# Fit and transform the vectorizor
cvec.fit(X_train)

Xcvec_train = cvec.transform(X_train)
Xcvec_test = cvec.transform(X_test)

# Fit the classifier
nb.fit(Xcvec_train,y_train)

# Create the predictions for Y training data
preds = nb.predict(Xcvec_test)
pre = nb.predict(cvec.transform(["Bombs fuck you"]))
print(nb.score(Xcvec_test, y_test),pre)