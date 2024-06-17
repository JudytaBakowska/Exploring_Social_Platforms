import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from .analysis import analyze_sentiment, analyze_words, analyze_text_length
from .reduction import pca_analysis, tsne_analysis, umap_analysis
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
import torch

nltk.download('stopwords')

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=1).tolist()
    
    # Convert predicted class to sentiment label
    sentiments = ['negative', 'neutral', 'positive']
    results = []
    for predicted_class in predicted_classes:
        if predicted_class == 0 or predicted_class == 1:
            results.append('negative')
        elif predicted_class == 2:
            results.append('neutral')
        else:
            results.append('positive')
    return results

def check_columns(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    return not bool(missing_columns)


def check_sentiment(dataframe):
    allowed_sentiments = {'positive', 'negative', 'neutral'}
    return dataframe['Sentiment'].isin(allowed_sentiments).all()


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def preprocess_dataframe(df):
    df['Text'] = df['Text'].apply(preprocess_text)
    return df

def load_and_analyze_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)

        if check_columns(df, ['Text', 'Sentiment']):
            if check_sentiment(df):
                st.success('CSV file meets the requirements.')
            else:
                st.error("Sentiment column must contain only 'positive', 'negative' or 'neutral' values.")

        elif check_columns(df, ['Text']):
            st.success('CSV file meets the requirements.')
            st.info("Sentiment column missing. Predicting sentiments...")

            # Ensure the 'Text' column contains strings
            df['Text'] = df['Text'].astype(str)

            # Predict sentiment for each row
            batch_size = 32  # Adjust batch size as needed
            num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
            sentiments = []

            for i in range(num_batches):
                batch_texts = df['Text'][i * batch_size: (i + 1) * batch_size].tolist()
                sentiments.extend(predict_sentiment(batch_texts))
                #st.info(f'Processed batch {i+1}/{num_batches}')

            df['Sentiment'] = sentiments
            
        else:
            st.error(
                "CSV file does not meet the requirements. It must contain at least 'Text' column.")

        with st.sidebar:
            selected = option_menu(
                "Menu",
                ["Home", "Mood Analysis", "Most Frequent Words", "Text Length Analysis", "Dimensionality Reduction"]
            )

        df = preprocess_dataframe(df)

        if selected == "Home":
            st.header(f"You chose {uploaded_file.name} file")
            st.dataframe(df)

        elif selected == "Mood Analysis":
            analyze_sentiment(df)

        elif selected == "Most Frequent Words":
            analyze_words(df)

        elif selected == "Text Length Analysis":
            analyze_text_length(df)

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

    except UnicodeDecodeError:
        st.error('Decoding Error: The file is not encoded as UTF-8. Try again with a UTF-8 encoded file.')
    except pd.errors.ParserError:
        st.error('Parsing Error: The CSV file may be corrupted or have an invalid format.')
    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')
