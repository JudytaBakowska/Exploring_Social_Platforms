import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import streamlit as st
import pandas as pd
from utils.plots import *
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack
import numpy as np
import os

def pca_analysis(df, csv_path='pca_data.csv'):
    if os.path.exists(csv_path):
        final_pca_df = pd.read_csv(csv_path)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        X_text = vectorizer.fit_transform(df['Text'])

        label_encoder = LabelEncoder()
        X_sentiment = label_encoder.fit_transform(df['Sentiment']).reshape(-1, 1)

        X_combined = hstack((X_text, X_sentiment))

        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_combined)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled.toarray())

        pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
        final_pca_df = pd.concat([pca_df, df[['Sentiment']]], axis=1)
        final_pca_df.to_csv(csv_path, index=False)

    st.subheader('PCA of Text Data')
    plot_pca(final_pca_df)


def tsne_analysis(df, csv_path='tsne_data.csv'):
    if os.path.exists(csv_path):
        final_tsne_df = pd.read_csv(csv_path)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        X_text = vectorizer.fit_transform(df['Text'])

        label_encoder = LabelEncoder()
        X_sentiment = label_encoder.fit_transform(df['Sentiment']).reshape(-1, 1)

        X_combined = hstack((X_text, X_sentiment))

        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_combined)

        tsne = TSNE(n_components=2, perplexity=50, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled.toarray())

        tsne_df = pd.DataFrame(data=X_tsne, columns=['Component 1', 'Component 2'])
        final_tsne_df = pd.concat([tsne_df, df[['Sentiment']]], axis=1)
        final_tsne_df.to_csv(csv_path, index=False)

    st.subheader('t-SNE of Text Data')
    plot_tsne(final_tsne_df)

def umap_analysis(df, csv_path='umap_data.csv'):
    if os.path.exists(csv_path):
        final_umap_df = pd.read_csv(csv_path)
    else:
        df.dropna(subset=['Text', 'Sentiment'], inplace=True)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        X_text = vectorizer.fit_transform(df['Text'])

        label_encoder = LabelEncoder()
        X_sentiment = label_encoder.fit_transform(df['Sentiment']).reshape(-1, 1)

        X_combined = hstack((X_text, X_sentiment))

        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_combined)

        umap_reducer = UMAP(n_components=2, n_neighbors=10, min_dist=0.05, random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled.toarray())

        umap_df = pd.DataFrame(data=X_umap, columns=['Component 1', 'Component 2'])
        final_umap_df = pd.concat([umap_df, df[['Sentiment']]], axis=1)
        final_umap_df.to_csv(csv_path, index=False)

    st.subheader('UMAP of Text Data')
    plot_umap(final_umap_df)
