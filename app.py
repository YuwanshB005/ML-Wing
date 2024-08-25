import streamlit as st
import re
import nltk
import pandas as pd
from googleapiclient.discovery import build
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Initialize the YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyD9pKknR7eJ0SAMzHQpkKbr5uB52EXv8yk"  # Replace with your actual YouTube API key
youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Function to fetch YouTube comments
def fetch_youtube_comments(video_id):
    """Fetch comments from a YouTube video using the video ID."""
    comments = []
    
    # Extract the actual video ID
    video_id = video_id.split('v=')[-1].split('&')[0]
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        response = request.execute()

        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['likeCount'],
                    comment['textOriginal'],
                    item['snippet']['isPublic']
                ])
            
            # Check if there is a next page of comments
            if 'nextPageToken' in response:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break
    except Exception as e:
        st.write(f"An error occurred while fetching comments: {e}")
    
    return comments

# Streamlit app layout
st.title('YouTube Comment Sentiment Analysis')

# Input for YouTube video ID
video_id = st.text_input("Enter YouTube video ID:")




# Fetch and analyze comments from a YouTube video
if st.button("Fetch Comments"):
    if video_id.strip():
        comments = fetch_youtube_comments(video_id)
        
        if comments:
            # Convert to DataFrame
            df_comments = pd.DataFrame(comments, columns=['author', 'published_at', 'like_count', 'text', 'public'])
            st.write("Fetched Comments:")
            st.dataframe(df_comments)
        else:
            st.write("No comments found or invalid video ID.")
    else:
        st.write("Please enter a valid YouTube video ID.")