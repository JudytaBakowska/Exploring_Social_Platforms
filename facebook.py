import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def facebook_read():
    df_facebook = pd.read_csv('datasets/facebook_dataset.csv')
    
    return df_facebook

def facebook_home():
    st.header('You choose Facebook Data dataset')


def facebook_gender(df_facebook):
    genders = df_facebook['gender'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(genders,labels = ['Male', 'Female'], autopct = '%.2f%%', colors = ['lightblue', 'pink'])
    ax.set_title('Female & Male')
    st.pyplot(fig) 
    
    
def facebook_age(df_facebook):
        
    option = st.radio(
        "Select the age group you want to see", #popraiwc
        ("By age group", "General"),
        index = None
    )
    
    if option == "By age group":
        ages = df_facebook['age'].value_counts().sort_values(na_position='first',ascending=False)
        
        option = st.selectbox(
        "Select the age group you want to see",
        ("11-20","21-30","31-40","41-50","51-60","61-70","71-80","81-90","91-100","100>"),
        index = None
        )
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightyellow', 'lightgrey', 'lightgoldenrodyellow', 'lightcyan', 'lightsalmon']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        selected_ages = None
        labels = None
        
        match option:
            case "11-20":
                selected_ages = ages[(ages.index>10) & (ages.index<=20)]
                labels = [str(age) for age in selected_ages.index]
            case "21-30":
                selected_ages = ages[(ages.index>20) & (ages.index<=30)]
                labels = [str(age) for age in selected_ages.index]
            case "31-40":
                selected_ages = ages[(ages.index>30) & (ages.index<=40)]
                labels = [str(age) for age in selected_ages.index]
            case "41-50":
                selected_ages = ages[(ages.index>40) & (ages.index<=50)]
                labels = [str(age) for age in selected_ages.index]
            case "51-60":
                selected_ages = ages[(ages.index>50) & (ages.index<=60)]
                labels = [str(age) for age in selected_ages.index]
            case "61-70":
                selected_ages = ages[(ages.index>60) & (ages.index<=70)]
                labels = [str(age) for age in selected_ages.index]
            case "71-80":
                selected_ages = ages[(ages.index>70) & (ages.index<=80)]
                labels = [str(age) for age in selected_ages.index]
            case "81-90":
                selected_ages = ages[(ages.index>80) & (ages.index<=90)]
                labels = [str(age) for age in selected_ages.index]
            case "91-100":
                selected_ages = ages[(ages.index>90) & (ages.index<=100)]
                labels = [str(age) for age in selected_ages.index]
            case "100>":
                selected_ages = ages[(ages.index>100)]
                labels = [str(age) for age in selected_ages.index]
                
        if selected_ages is not None and labels is not None:        
            ax.pie(selected_ages,labels = labels, autopct = '%.2f%%',colors=colors )
            ax.set_title('Age Distribution')
            st.pyplot(fig) 
        
    elif option == "General":
        fig,ax = plt.subplots()
        sns.histplot(data = df_facebook, x = 'age', bins = 30, kde = True, ax = ax)
        ax.set_title('Age Distribution of Facebook Users')
        st.pyplot(fig)
        
        
    

