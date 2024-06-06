import matplotlib.pyplot as plt
import streamlit as st


def plot_sorted_bar_chart(word_counts, color, title):
    most_common_words = word_counts.most_common(20)
    words, counts = zip(*most_common_words)

    fig, ax = plt.subplots()
    ax.barh(words, counts, color=color)
    ax.set_xlabel('Counts')
    ax.set_title(title)
    ax.invert_yaxis()

    st.pyplot(fig)


def plot_bar_chart(data, title, ylabel):
    fig, ax = plt.subplots()
    sentiments = list(data.keys())
    values = list(data.values())
    colors = ['green', 'red', 'blue']

    ax.bar(sentiments, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Sentiment')

    st.pyplot(fig)
