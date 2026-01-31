import os
from googleapiclient.discovery import build

API_KEY = os.getenv("YOUTUBE_API_KEY")

def fetch_youtube_comments(query, max_comments=30):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    # Search videos
    search_response = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=3
    ).execute()

    comments = []

    for item in search_response["items"]:
        video_id = item["id"]["videoId"]

        comment_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=20,
            textFormat="plainText"
        ).execute()

        for c in comment_response["items"]:
            text = c["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if len(text) > 15:
                comments.append(text)

            if len(comments) >= max_comments:
                return comments

    return comments
