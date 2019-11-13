
from flask import Flask, redirect, url_for, request , render_template
from bs4 import BeautifulSoup
import csv 
import requests 
from psaw import PushshiftAPI
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

flag = True
nb = MultinomialNB(alpha = 0.36)
cvec = CountVectorizer(ngram_range= (1, 3))

app = Flask(__name__) 

def scrape_data(subreddit):
    print("cp1")
    api = PushshiftAPI()
    print("cp2")
    # Create list of scraped data
    scrape_list = list(api.search_submissions(subreddit=subreddit,
                                filter=['title', 'subreddit', 'num_comments', 'author', 'subreddit_subscribers', 'score', 'domain', 'created_utc'],
                                limit=15000))
    print("cp3")
    clean_scrape_lst = []
    for i in range(len(scrape_list)):
        scrape_dict = {}
        scrape_dict['subreddit'] = scrape_list[i][5]#Name of subreddit
        scrape_dict['author'] = scrape_list[i][0]
        scrape_dict['domain'] = scrape_list[i][2]#Publishing House 
        scrape_dict['title'] = scrape_list[i][7]
        scrape_dict['num_comments'] = scrape_list[i][3]
        scrape_dict['score'] = scrape_list[i][4]#upvotes-downvotes
        scrape_dict['timestamp'] = scrape_list[i][1]#time in epoch format
        clean_scrape_lst.append(scrape_dict)
    print("cp4")
    # Show number of subscribers
    print(subreddit, 'subscribers:',scrape_list[1][6])
    
    # Return list of scraped data
    return clean_scrape_lst

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

@app.route('/') 
def main(): 
    return render_template("main.html")

@app.route('/get_data')
def get_data():
    print("Hello")
    df_not_onion = pd.DataFrame(scrape_data('nottheonion'))
    # Save data to csv
    df_not_onion.to_csv('not_onion.csv')
    print(f'df_not_onion shape: {df_not_onion.shape}')
    #print(df_not_onion.head())
    df_onion = pd.DataFrame(scrape_data('theonion'))
    df_onion.to_csv('the_onion.csv')
    print(f'df_onion shape: {df_onion.shape}')
    #df_onion.head()

    return {"ID": "Successfully scraped!"}
    #return render_template("access.html")   

@app.route('/train')
def train():
    global flag
    global nb
    global cvec
    
    flag = False
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

    #Instantiate the classifier and vectorizer
    #nb = MultinomialNB(alpha = 0.36)
    #cvec = CountVectorizer(ngram_range= (1, 3))

    # Fit and transform the vectorizor
    cvec.fit(X_train)

    Xcvec_train = cvec.transform(X_train)
    Xcvec_test = cvec.transform(X_test)

    # Fit the classifier
    nb.fit(Xcvec_train,y_train)

    # Create the predictions for Y training data
    preds = nb.predict(Xcvec_test)
    print(nb.score(Xcvec_test, y_test))

    return {"ID": str(nb.score(Xcvec_test, y_test)*100)+"% Accuracy"}
    #return render_template("access.html")   

@app.route('/predict', methods=['GET','POST'])
def predict():
    global nb
    global flag
    global cvec

    if flag:
        return {"ID": "Train model first!"}

    query=None
    pre=""
    if request.method == "POST":
        query=request.form['data']
    print(query)
    if query is not None:
        pre = nb.predict(cvec.transform([query]))
        print("Prediction!! - {}".format(pre))
    pre = (pre == [0]) and "Legit!" or "False!"    
    return {"ID": pre}

        
  
if __name__ == '__main__': 
   app.run(debug = True) 
