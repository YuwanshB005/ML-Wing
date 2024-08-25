from googleapiclient.discovery import build
import streamlit as st

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "YOUR_API_KEY_HERE"
youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

def fetch_youtube_comments(video_id):
    comments = []
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
