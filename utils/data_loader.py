import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from .analysis import analyze_sentiment, analyze_words, analyze_text_length


def check_columns(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    return not bool(missing_columns)


def check_sentiment(dataframe):
    allowed_sentiments = {'positive', 'negative', 'neutral'}

    if not dataframe['Sentiment'].isin(allowed_sentiments).all():
        return False

    return True


def load_and_analyze_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)

        if check_columns(df, ['Text', 'Sentiment']) and check_sentiment(df):
            if check_sentiment(df):
                st.success('CSV file meets the requirements.')
            else:
                st.error("Sentiment column must contain only 'positive', 'negative' or 'neutral' values.")

        elif check_columns(df, ['Text']):
            st.success('CSV file meets the requirements.')
            # TODO: Uzupełnić kolumnę 'Sentiment' używając modelu ML

        else:
            st.error(
                "CSV file does not meet the requirements. It must contain at least 'Text' and 'Sentiment' columns.")

        with st.sidebar:
            selected = option_menu(
                "Menu",
                ["Home", "Mood Analysis", "Most Frequent Words", "Text Length Analysis"]
            )

        if selected == "Home":
            st.header(f"You chose {uploaded_file.name} file")
            st.dataframe(df)

        elif selected == "Mood Analysis":
            analyze_sentiment(df)

        elif selected == "Most Frequent Words":
            analyze_words(df)

        elif selected == "Text Length Analysis":
            analyze_text_length(df)

    except UnicodeDecodeError:
        st.error('Decoding Error: The file is not encoded as UTF-8. Try again with a UTF-8 encoded file.')
    except pd.errors.ParserError:
        st.error('Parsing Error: The CSV file may be corrupted or have an invalid format.')
    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')
