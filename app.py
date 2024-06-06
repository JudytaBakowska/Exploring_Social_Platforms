from twitter import *


def check_columns(dataframe, required_columns):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        st.warning(f"The CSV file must contain columns: {', '.join(required_columns)}")
        return False
    return True


st.title('Interactive Dashboard for Social Platform Exploration')
st.text('This is an interactive dashboard to analyze various social media datasets')

uploaded_file = st.file_uploader('Upload your CSV file here', type='csv')


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if check_columns(df, ['Text']) or check_columns(df, ['Text', 'Sentiment']):
            st.success('CSV file meets the requirements.')
            st.dataframe(df)

            if 'Sentiment' in df.columns:
                sentiment_count = df['Sentiment'].value_counts()
                st.bar_chart(sentiment_count)
        else:
            st.error('CSV file does not contain required columns.')
    except UnicodeDecodeError:
        st.error(
            'Decoding Error: The file is not encoded as UTF-8. Try again with a UTF-8 encoded file.')
    except pd.errors.ParserError:
        st.error('Parsing Error: The CSV file may be corrupted or have an invalid format.')
    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')
