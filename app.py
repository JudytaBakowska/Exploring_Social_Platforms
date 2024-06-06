import streamlit as st
from utils.data_loader import load_and_analyze_csv


def main():
    st.title('Interactive Dashboard for Social Platform Exploration')
    st.text('This is an interactive dashboard to analyze various social media datasets')

    uploaded_file = st.file_uploader('Upload your CSV file here', type='csv')

    if uploaded_file:
        load_and_analyze_csv(uploaded_file)


if __name__ == "__main__":
    main()
