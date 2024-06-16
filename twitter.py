from collections import Counter
import streamlit as st
import pandas as pd
from twitter import *
from streamlit_option_menu import option_menu
from facebook import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.data_loader import preprocess_dataframe

def twitter_read():
    df_twitter = pd.read_csv('datasets/twitter_dataset.csv', encoding='ISO-8859-1')
    return df_twitter


def twitter_home():
    st.header('You choose sentiment140 dataset with tweets')

    st.markdown('''
        ***This is the sentiment140 dataset.***\n
        It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment .\n
        It contains the following 6 fields:

        ***target***: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)\n
        ***ids***: The id of the tweet ( 2087) \n
        ***date***: the date of the tweet (Sat May 16 23:58:44 UTC 2009)\n
        ***flag***: The query (lyx). If there is no query, then this value is NO_QUERY.\n
        ***user***: the user that tweeted (robotickilldozr)\n
        ***text***: the text of the tweet (Lyx is cool)''')


df_twitter = twitter_read()
df_twitter.columns = ['Target', 'IDs', 'Date', 'Flag', 'User', 'Text', 'Sentiment']
df_twitter = preprocess_dataframe(df_twitter)  # Usuń stop words
negative_tweets = df_twitter[df_twitter['Target'] == 0]
positive_tweets = df_twitter[df_twitter['Target'] == 4]


def twitter_moods(df_twitter):
    st.header("Sentiment analysis")
    df_twitter['Date'] = pd.to_datetime(df_twitter['Date'])

    df_twitter.set_index('Date', inplace=True)

    negative_tweets = df_twitter[df_twitter['Target'] == 0].resample('D').size()
    positive_tweets = df_twitter[df_twitter['Target'] == 4].resample('D').size()

    option = st.selectbox(
        "What mood you would like to see on the plot?",
        ("Negative", "Positive", "Mixed"),
        index=None,
        placeholder="Select the mood you want to see... "
    )

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    if option == "Negative":
        negative_tweets.plot(ax=ax, color='red')
    elif option == "Positive":
        positive_tweets.plot(ax=ax, color='green')
    else:
        negative_tweets.plot(ax=ax, color='red')
        positive_tweets.plot(ax=ax, color='green')

    ax.set_xlabel('Date')
    ax.set_ylabel(f'Counts of {option} moods')
    ax.set_title('Changes in sentiment numbers ')
    st.pyplot(fig)


def twitter_words(df_twitter):
    st.header("Words analysis")
    positive = positive_tweets['Text']
    negative = negative_tweets['Text']

    positive_words = ' '.join(positive).split()
    positive_words_counts = Counter(word for word in positive_words if len(word) > 4)

    negative_words = ' '.join(negative).split()
    negative_words_counts = Counter(word for word in negative_words if len(word) > 4)

    words = ' '.join(df_twitter['Text']).split()
    words_counts = Counter(word for word in words if len(word) > 4)

    option = st.radio(
        "Top 20 words used in tweets",
        [":red[Negative]", ":green[Positive]", "General"],
        index=None,
    )
    if option == ":red[Negative]":
        st.write(f"Most frequent words in negative tweets: {negative_words_counts.most_common(10)}")
    elif option == ":green[Positive]":
        st.write(f"Most frequent words in positive tweets: {positive_words_counts.most_common(10)}")
    else:
        st.write(f"Most frequent words in tweets: {words_counts.most_common(10)}")


def twitter_users(df_twitter):
    st.header("Users analysis")
    user_counts = df_twitter['User'].value_counts()
    st.write(f"Most active users: {user_counts[:15]}")

    fig, ax = plt.subplots()
    user_counts[:15].plot(kind='bar', ax=ax)
    ax.set_title("Most active users")
    st.pyplot(fig)
