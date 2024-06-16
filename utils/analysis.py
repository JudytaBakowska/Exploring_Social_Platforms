import streamlit as st
from collections import Counter
from matplotlib import pyplot as plt
from .plots import plot_sorted_bar_chart, plot_bar_chart, plot_reduction, plot_umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack
import pandas as pd

nltk.download('stopwords')

def analyze_sentiment(df):
    st.header("Sentiment Analysis")
    sentiment_count = df['Sentiment'].value_counts()
    st.bar_chart(sentiment_count)

    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_count, labels=sentiment_count.index, autopct='%1.1f%%')
    ax1.axis('equal')
    ax1.legend(sentiment_count.index, loc="best")

    st.pyplot(fig1)


def analyze_words(df):
    st.header("Words Analysis")

    positive_texts = df[df['Sentiment'] == 'positive']['Text']
    negative_texts = df[df['Sentiment'] == 'negative']['Text']
    neutral_texts = df[df['Sentiment'] == 'neutral']['Text']

    positive_words_counts = Counter(word for word in ' '.join(positive_texts).split() if len(word) > 4)
    negative_words_counts = Counter(word for word in ' '.join(negative_texts).split() if len(word) > 4)
    neutral_words_counts = Counter(word for word in ' '.join(neutral_texts).split() if len(word) > 4)
    all_words_counts = Counter(word for word in ' '.join(df['Text']).split() if len(word) > 4)

    option = st.radio(
        "Top 20 words used in tweets",
        [":red[Negative]", ":green[Positive]", ":blue[Neutral]", "General"],
        index=None,
    )

    if option == ":red[Negative]":
        plot_sorted_bar_chart(negative_words_counts, 'red', 'Top 20 Words in Negative Tweets')
    elif option == ":green[Positive]":
        plot_sorted_bar_chart(positive_words_counts, 'green', 'Top 20 Words in Positive Tweets')
    elif option == ":blue[Neutral]":
        plot_sorted_bar_chart(neutral_words_counts, 'blue', 'Top 20 Words in Neutral Tweets')
    else:
        plot_sorted_bar_chart(all_words_counts, 'black', 'Top 20 Words in All Tweets')


def analyze_text_length(df):
    st.header("Text Length Analysis")

    df['Text Length'] = df['Text'].apply(len)
    avg_length_positive = df[df['Sentiment'] == 'positive']['Text Length'].mean()
    avg_length_negative = df[df['Sentiment'] == 'negative']['Text Length'].mean()
    avg_length_neutral = df[df['Sentiment'] == 'neutral']['Text Length'].mean()

    avg_lengths = {
        'Positive': avg_length_positive,
        'Negative': avg_length_negative,
        'Neutral': avg_length_neutral
    }

    plot_bar_chart(avg_lengths, 'Average Text Length by Sentiment', 'Average Text Length')


# def pca_analysis(df):
#     # Wektoryzacja danych tekstowych za pomocą TF-IDF
#     vectorizer = TfidfVectorizer(stop_words = 'english', max_features=3000)
#     X_text = vectorizer.fit_transform(df['Text'])

#     label_encoder = LabelEncoder()
#     X_sentiment = label_encoder.fit_transform(df['Sentiment']).reshape(-1, 1)

#     X_combined = hstack((X_text, X_sentiment))
    
#     scaler = StandardScaler(with_mean=False)  # StandardScaler nie wspiera sparse matrices z with_mean=True
#     X_scaled = scaler.fit_transform(X_combined)

#     # Wyciągnij kolumnę z sentymentem
#     #y = df['Sentiment'].values
    
#     # Zastosuj PCA
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled.toarray())

#     pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])

#     final_df = pd.concat([pca_df, df[['Sentiment']]], axis=1)
#     st.subheader('PCA of Text Data - dataset')
#     final_df.head()


#     # Wizualizacja PCA
#     st.subheader('PCA of Text Data')
#     plt.figure(figsize=(10, 7))
#     colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
#     plt.scatter(final_df['Principal Component 1'], final_df['Principal Component 2'], 
#                 c=final_df['Sentiment'].apply(lambda x: colors[x]), alpha=0.5)
#     plt.title('PCA of Text Data')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=label) 
#                         for label in colors.keys()], title="Sentiment")
#     plt.grid(True)
#     plt.show()
    


# def tsne_analysis(df):
#     # Wektoryzacja danych tekstowych za pomocą TF-IDF
#     vectorizer = TfidfVectorizer(max_features=300)
#     X = vectorizer.fit_transform(df['Text']).toarray()
    
#     # Wyciągnij kolumnę z sentymentem
#     y = df['Sentiment'].values
    
    
#     # Zastosuj t-SNE
#     tsne = TSNE(n_components=2, random_state=42)
#     X_tsne = tsne.fit_transform(X)
    

#     # Wizualizacja t-SNE
#     st.subheader('t-SNE of Text Data')
#     plot_reduction(X_tsne, y, 't-SNE of Text Data')




# def umap_analysis(df):
#     # Wektoryzacja danych tekstowych za pomocą TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
#     X_text = vectorizer.fit_transform(df['Text'])

#     # Kodowanie kolumny sentymentu
#     label_encoder = LabelEncoder()
#     X_sentiment = label_encoder.fit_transform(df['Sentiment']).reshape(-1, 1)

#     # Połączenie danych
#     X_combined = hstack((X_text, X_sentiment))

#     # Standaryzacja danych
#     scaler = StandardScaler(with_mean=False)  # StandardScaler nie wspiera sparse matrices z with_mean=True
#     X_scaled = scaler.fit_transform(X_combined)

#     # Zastosowanie UMAP
#     umap_reducer = UMAP(n_components=2, random_state=42)
#     X_umap = umap_reducer.fit_transform(X_scaled.toarray())

#     umap_df = pd.DataFrame(data=X_umap, columns=['Component 1', 'Component 2'])
#     final_umap_df = pd.concat([umap_df, df[['Sentiment']]], axis=1)

#     st.subheader('UMAP of Text Data')
#     plot_umap(final_umap_df)