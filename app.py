# DEVELOPER_KEY = "AIzaSyD9pKknR7eJ0SAMzHQpkKbr5uB52EXv8yk"  # Replace with your actual YouTube API key
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_resources()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# YouTube API setup
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyD9pKknR7eJ0SAMzHQpkKbr5uB52EXv8yk"  # Replace with your actual YouTube API key

@st.cache_resource
def get_youtube_client():
    return build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

youtube = get_youtube_client()

# Text cleaning functions
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Load Tokenizer and Model
@st.cache_resource
def load_models():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('samarth.pkl', 'rb') as f:
        model = pickle.load(f)
    return tokenizer, model

tokenizer, model = load_models()

# Predict Sentiment
@st.cache_data
def predict_sentiments(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=200)
    predictions = model.predict(padded_sequences)
    return ["positive" if pred > 0.5 else "negative" for pred in predictions]

# Fetch YouTube comments
def fetch_youtube_comments(video_url, max_comments=500):
    video_id = video_url.split('v=')[-1].split('&')[0]
    comments = []
    
    try:
        next_page_token = None
        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['likeCount'],
                    comment['textOriginal'],
                    item['snippet']['isPublic']
                ])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    except HttpError as e:
        st.error(f"An HTTP error occurred: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
    
    return comments

# Process comments and predict sentiment
def process_comments_and_predict(video_url, max_comments=500):
    comments = fetch_youtube_comments(video_url, max_comments)
    if comments is None:
        return None, None

    df_comments = pd.DataFrame(comments, columns=['author', 'published_at', 'like_count', 'text', 'public'])
    
    with st.spinner("Cleaning and processing comments..."):
        df_comments['cleaned_text'] = df_comments['text'].apply(clean_text)
    
    df_comments = df_comments[df_comments['cleaned_text'] != ''].reset_index(drop=True)
    
    with st.spinner("Predicting sentiment..."):
        df_comments['sentiment'] = predict_sentiments(df_comments['cleaned_text'].tolist())

    df_positive = df_comments[df_comments['sentiment'] == 'positive']
    df_negative = df_comments[df_comments['sentiment'] == 'negative']
    
    return df_positive, df_negative

# Streamlit app
def main():
    st.title('YouTube Comment Sentiment Analysis')

    video_url = st.text_input("Enter YouTube video URL:")
    max_comments = st.slider("Maximum number of comments to analyze", 1, 1000, 500)

    if st.button("Analyze Comments"):
        if video_url.strip():
            with st.spinner("Fetching and analyzing comments..."):
                df_positive, df_negative = process_comments_and_predict(video_url, max_comments)
            
            if df_positive is not None and df_negative is not None:
                st.success("Analysis complete!")
                
                st.subheader("Positive Comments")
                st.dataframe(df_positive[['author', 'text', 'sentiment']])
                
                st.subheader("Negative Comments")
                st.dataframe(df_negative[['author', 'text', 'sentiment']])
                
                st.write(f"Total comments analyzed: {len(df_positive) + len(df_negative)}")
                st.write(f"Positive comments: {len(df_positive)}")
                st.write(f"Negative comments: {len(df_negative)}")
            else:
                st.error("Failed to analyze comments. Please check the error message above.")
        else:
            st.warning("Please enter a valid YouTube video URL.")

if __name__ == "__main__":
    main()