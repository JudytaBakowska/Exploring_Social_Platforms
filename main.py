import streamlit as st
import pandas as pd
from twitter import *
from streamlit_option_menu import option_menu
from facebook import *

import matplotlib.pyplot as plt

twitter_df = twitter_read()
facebook_df = facebook_read()

st.title('Interactive Dashboard for Social Platform Exploration')
st.text('This is an interactive dashboard to analyse various social media dataset')

uploaded_file = st.file_uploader('Upload your file here')

if uploaded_file:
    df = pd.read_csv(uploaded_file,encoding='ISO-8859-1')

    if twitter_df.equals(df):
        df.columns = ['Target','IDs','Date','Flag','User','Text']
        
        with st.sidebar:
            selected = option_menu(
            "Menu",
            ("Home", "Mood Analysis", "Most frequent words", "Most active users")
        )
        if selected == "Home":
            twitter_home()
        elif selected == "Mood Analysis":
            twitter_moods(df)
        elif selected == "Most frequent words":
            twitter_words(df)
        elif selected == "Most active users":
            twitter_users(df)
            
    if facebook_df.equals(df):
        df.dropna(inplace=True)
    
        with st.sidebar:
            selected = option_menu(
                "Menu",
                ("Home","Gender Analysis", "Age Analysis")
            )
            
        if selected == "Home":
            facebook_home()
        elif selected == "Gender Analysis":
            facebook_gender(df)
        elif selected == "Age Analysis":
            facebook_age(df)
  
  