import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

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


def plot_pca(final_pca_df):
    plt.figure(figsize=(10, 7))
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
    plt.scatter(final_pca_df['Principal Component 1'], final_pca_df['Principal Component 2'], 
                c=final_pca_df['Sentiment'].apply(lambda x: colors[x]), alpha=0.5)
    plt.title('PCA of Text Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=label) 
                        for label in colors.keys()], title="Sentiment")
    plt.grid(True)
    st.pyplot(plt.gcf())

def plot_tsne(final_tsne_df):
    plt.figure(figsize=(10, 7))
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
    plt.scatter(final_tsne_df['Component 1'], final_tsne_df['Component 2'], 
                c=final_tsne_df['Sentiment'].apply(lambda x: colors[x]), alpha=0.5)
    plt.title('t-SNE of Text Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=label) 
                        for label in colors.keys()], title="Sentiment")
    plt.grid(True)
    st.pyplot(plt.gcf())

def plot_umap(final_umap_df):
    plt.figure(figsize=(10, 7))
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
    plt.scatter(final_umap_df['Component 1'], final_umap_df['Component 2'], 
                c=final_umap_df['Sentiment'].apply(lambda x: colors[x]), alpha=0.5)
    plt.title('UMAP of Text Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=label) 
                        for label in colors.keys()], title="Sentiment")
    plt.grid(True)
    st.pyplot(plt.gcf())