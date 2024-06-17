import streamlit as st
import pandas as pd
from twitter import *
from streamlit_option_menu import option_menu
from facebook import *
from utils.reduction import *
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords



twitter_df = twitter_read()
facebook_df = facebook_read()

st.title('Interactive Dashboard for Social Platform Exploration')
st.text('This is an interactive dashboard to analyse various social media dataset')

uploaded_file = st.file_uploader('Upload your file here')

def clean_data(df):
    st.write("Initial data shape:", df.shape)
    df = df.dropna(subset=['Text', 'Sentiment'])
    df = df[df['Text'].str.strip() != '']
    df = df[df['Sentiment'].str.strip() != '']
    st.write("Data shape after cleaning:", df.shape)
    return df


if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # if twitter_df.equals(df):
    if set(twitter_df.columns) == set(df.columns):
        df.columns = ['Target', 'IDs', 'Date', 'Flag', 'User', 'Text', 'Sentiment']

        with st.sidebar:
            selected = option_menu(
                "Menu",
                ("Home", "Mood Analysis", "Most frequent words", "Most active users", "Dimensionality Reduction")
            )
        if selected == "Home":
            twitter_home()
        elif selected == "Mood Analysis":
            twitter_moods(df)
        elif selected == "Most frequent words":
            twitter_words(df)
        elif selected == "Most active users":
            twitter_users(df)
        elif selected == "Dimensionality Reduction":
            reduction_option = option_menu(
                "Reduction Method",
                ["PCA", "UMAP", "t-SNE"],
                icons=["box", "scatter-plot", "chart-bar"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal"
            )

            if reduction_option == "PCA":
                pca_analysis(df, csv_path=uploaded_file.name.replace('.csv', '_pca_data.csv'))
            elif reduction_option == "UMAP":
                umap_analysis(df, csv_path=uploaded_file.name.replace('.csv', '_umap_data.csv'))
            elif reduction_option == "t-SNE":
                tsne_analysis(df, csv_path=uploaded_file.name.replace('.csv', '_tsne_data.csv'))

    if facebook_df.equals(df):
        df.dropna(inplace=True)

        with st.sidebar:
            selected = option_menu(
                "Menu",
                ("Home", "Gender Analysis", "Age Analysis", "Dimensionality Reduction")
            )

        if selected == "Home":
            facebook_home()
        elif selected == "Gender Analysis":
            facebook_gender(df)
        elif selected == "Age Analysis":
            facebook_age(df)
        elif selected == "Dimensionality Reduction":
            reduction_option = option_menu(
                "Reduction Method",
                ["PCA", "UMAP", "t-SNE"],
                icons=["box", "scatter-plot", "chart-bar"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal"
            )

            if reduction_option == "PCA":
                pca_analysis(df, csv_path=uploaded_file.name.replace('.csv', '_pca_data.csv'))
            elif reduction_option == "UMAP":
                umap_analysis(df, csv_path=uploaded_file.name.replace('.csv', '_umap_data.csv'))
            elif reduction_option == "t-SNE":
                tsne_analysis(df, csv_path=uploaded_file.name.replace('.csv', '_tsne_data.csv'))

  