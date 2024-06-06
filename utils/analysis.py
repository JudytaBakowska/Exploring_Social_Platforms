import streamlit as st
from collections import Counter
from matplotlib import pyplot as plt
from .plots import plot_sorted_bar_chart, plot_bar_chart


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
