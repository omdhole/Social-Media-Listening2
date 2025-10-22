import streamlit as st
import requests
import openai
import json
import plotly.express as px
import requests
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from youtube_search import YoutubeSearch
from youtube_comment_downloader import YoutubeCommentDownloader

# Use markdown + HTML for colored title - Added
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.markdown("""<style>.stApp{background:linear-gradient(135deg,#141E30, #243B55);}</style> """,unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='color: #FFD700; text-align:center; font-weight: bold; margin-top:30px;'>🎬 Sentiment Dashboard</h1>
    """,
    unsafe_allow_html=True
)

#OMDB_API_KEY = "23d1de4"

OMDB_API_KEY = st.secrets["api_keys"]["OMDB_API_KEY"]
#LLM_TOKEN = st.secrets["api_keys"]["LLM_TOKEN"]

LLM_ENDPOINT = "https://models.github.ai/inference"
MODEL_NAME = "openai/gpt-4.1"

def fetch_omdb_data(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    #st.write(data)
    if data.get("Response") == "True":
        return data
    else:
        return None
    
# KPI colour change - Added
def uniform_metric(label, value, color="#FFD700"):
    st.markdown(f"""
        <div style='
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,215,0,0.4);
            border-radius: 10px;
            padding: 20px 15px;
            text-align: center;
            box-shadow: 0 0 10px rgba(255, 215,0,0.08);
            margin:8px 0;
            width:220px;
            display:inline-block;
            '>
            <div style='color: {color}; font-weight: 700; font-size: 16px;'>{label}</div>
            <div style="color:#FFFFFF; 
            font-weight:600; font-size:20px;">{value}</div>
        </div>
    """, unsafe_allow_html=True)
    
# Use session state to store if movie was searched and the movie title
if 'movie_searched' not in st.session_state:
    st.session_state.movie_searched = False
if 'movie_title' not in st.session_state:
    st.session_state.movie_title = ""
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'overview'

# --- Main UI ---
st.markdown("""
            <style>
            .stTextInput> div>div>input {
            background-color:rgba(0,0,0,0.4);
            color:#FFFFFF;
            border:1px solid #FFD700;
            border-radius:10px;
            padding: 10px;
            font-size:16px;
            }
            ::placeholder{
            color:#cccccc;
            }
            .stButton>button{
            background-color: #FFD700;
            color: black;
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 10px 18 px;
            transition: all 0.3s ease;
            }
            .stButton>button:hover{
            background-color:#ffec80;
            box-shadow: 0 0 10px #FFD700;
            transform: scale(1.05);
            }
            </style>
            """,unsafe_allow_html=True)
st.markdown("<div style='margin-top:30px'></div>",unsafe_allow_html=True)
col1, col2=st.columns([8,1])
with col1:
    title_input = st.text_input("",placeholder="🔎 Enter movie title...",label_visibility="collapsed")
with col2:
    search_clicked = st.button("Search")

if search_clicked:
    if not title_input.strip():
        st.warning("Please enter title.")
    else:
        st.session_state.movie_searched = True
        st.session_state.movie_title = title_input.strip()
        st.session_state.selected_tab = 'overview'  # reset to overview on new search

if st.session_state.movie_searched:
    # Fetch data once here using st.session_state.movie_title
    omdb_data = fetch_omdb_data(st.session_state.movie_title)
    if omdb_data is None:
        st.error("Movie not found in OMDb. Please try another title.")
    
    else:
        st.markdown("---")
        tab1, tab2 = st.tabs([":material/dashboard: Overview & Performance" , ":material/insights: Social Media Insights"])

        st.markdown("""
        <style>
        /* --- Tab Container --- */
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            gap: 10px;
            justify-content: center;
        }

        /* --- Tabs (default & hover) --- */
        .stTabs [data-baseweb="tab"] {
            background-color: #ccc;       /* unselected tab background */
            color: #333;                  /* unselected text */
            padding: 8px 18px;
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: background-color 0.25s, color 0.25s, box-shadow 0.25s;
            cursor: pointer;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #bfbfbf;
            color: #000;
        }

        /* --- Selected Tab --- */
        .stTabs [aria-selected="true"] {
            background-color: #FFD700 !important;
            color: #fff !important;
            border-bottom: 3px solid #B19CD9 !important;
            box-shadow: 0 2px 6px rgba(108,99,255,0.3);
        }
        </style>
        """, unsafe_allow_html=True)

            
        with tab1:
            # --- Top metrics line ---
            col_sp1, col1, col2, col3, col_sp2 = st.columns([1, 2, 2, 2, 1])

            with col1:
                try:
                    uniform_metric("IMDb Rating", omdb_data.get("imdbRating", "N/A"))
                except Exception as e:
                    st.error(f"Error displaying IMDb Rating: {e}")
            with col2:
                try:
                    uniform_metric("Runtime (min)", omdb_data.get("Runtime", "N/A"))
                except Exception as e:
                    st.error(f"Error displaying Runtime: {e}")
            with col3:
                try:
                    uniform_metric("Total Seasons", omdb_data.get('totalSeasons', 'N/A'))
                except Exception as e:
                    st.error(f"Error displaying Total Seasons: {e}")
            
            # --- Styles ---
            st.markdown("""
            <style>
            .overview-card {
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,215,0,0.25);
                border-radius: 16px;
                padding: 25px 30px;
                margin-top: 20px;
                box-shadow: 0 0 25px rgba(0,0,0,0.5),
                            inset 0 0 20px rgba(255,215,0,0.05);
                backdrop-filter: blur(8px);
                color: #E6E6E6;
                line-height: 1.7;
                font-size: 18px;
            }
            .overview-title {
                color: #FFD700;
                font-size: 22px;
                font-weight: 800;
                text-transform: uppercase;
                margin-bottom: 15px;
                text-align: left;
                border-bottom: 1px solid rgba(255,215,0,0.3);
                padding-bottom: 8px;
            }
            .overview-layout {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: flex-start;
            }
            .overview-left {
                flex: 2;
                min-width: 280px;
            }
            .overview-right {
                flex: 1;
                text-align: center;
            }
            .overview-right img {
                width: 250px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(255,215,0,0.2);
            }
            .plot-text {
                margin-top: 15px;
                border-top: 1px solid rgba(255,215,0,0.2);
                padding-top: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

            poster = omdb_data.get("Poster", "N/A")
            poster_html = f'<img src="{poster}" alt="Poster"/>' if poster != "N/A" else "<p>No image available</p>"
            try:
                poster = omdb_data.get("Poster", "N/A")
                poster_html = f'<img src="{poster}" alt="Poster"/>' if poster != "N/A" else "<p>No image available</p>"
            except Exception as e:
                poster_html = "<p>Failed to load poster image.</p>"
                st.error(f"Error retrieving poster: {e}")

            # st.markdown(f"""
            # <div class="overview-card">
            # <div class="overview-title">Movie Overview</div>
            # <div class="overview-layout">
            #     <div class="overview-left">
            #     <p><b>Release Date:</b> {omdb_data.get('Released', 'N/A')}</p>
            #     <p><b>Genre:</b> {omdb_data.get('Genre', 'N/A')}</p>
            #     <p><b>Director:</b> {omdb_data.get('Director', 'N/A')}</p>
            #     <p><b>Writers:</b> {omdb_data.get('Writer', 'N/A')}</p>
            #     <p><b>Actors:</b> {omdb_data.get('Actors', 'N/A')}</p>
            #     <p><b>Language:</b> {omdb_data.get('Language', 'N/A')}</p>
            #     <p><b>Awards:</b> {omdb_data.get('Awards', 'N/A')}</p>
            #     </div>
            #     <div class="overview-right">{poster_html}</div>
            # </div>
            # <div class="plot-text"><b>Plot:</b> {omdb_data.get('Plot', 'N/A')}</div>
            # </div>
            # """, unsafe_allow_html=True)

            try:
                st.markdown(f"""
                <div class="overview-card">
                    <div class="overview-title">Movie Overview</div>
                    <div class="overview-layout">
                        <div class="overview-left">
                            <p><b>Release Date:</b> {omdb_data.get('Released', 'N/A')}</p>
                            <p><b>Genre:</b> {omdb_data.get('Genre', 'N/A')}</p>
                            <p><b>Director:</b> {omdb_data.get('Director', 'N/A')}</p>
                            <p><b>Writers:</b> {omdb_data.get('Writer', 'N/A')}</p>
                            <p><b>Actors:</b> {omdb_data.get('Actors', 'N/A')}</p>
                            <p><b>Language:</b> {omdb_data.get('Language', 'N/A')}</p>
                            <p><b>Awards:</b> {omdb_data.get('Awards', 'N/A')}</p>
                        </div>
                        <div class="overview-right">{poster_html}</div>
                    </div>
                    <div class="plot-text"><b>Plot:</b> {omdb_data.get('Plot', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to load movie overview: {e}")

            st.markdown("---")

            st.markdown(
                """
                <div style="
                    background: rgba(255,255,255,0.07);
                    border: 1px solid rgba(255,215,0,0.25);
                    border-radius: 12px;
                    padding: 10px 0;
                    text-align: center;
                    color: #FFD700;
                    font-weight: 700;
                    font-size: 18px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.3);
                ">
                    📝 Score Card
                </div>
                <div style="height: 10px;"></div>  <!-- Small spacer -->
                """,
                unsafe_allow_html=True
            )
            # #--- Score card block ---
            # Ratings = omdb_data.get("Ratings", [])

            # if Ratings:
            #     # Create columns dynamically
            #     cols = st.columns(len(Ratings))
            #     for col, rating in zip(cols, Ratings):
            #         source = rating.get("Source", "Unknown")
            #         value = rating.get("Value", "N/A")
            #         col.markdown(
            #             f"""
            #             <div style="
            #                 background: rgba(255,255,255,0.07);
            #                 border: 1px solid rgba(255,215,0,0.25);
            #                 border-radius: 12px;
            #                 padding: 15px;
            #                 text-align: center;
            #                 box-shadow: 0 0 15px rgba(0,0,0,0.3);
            #             ">
            #                 <div style="color: #FFD700; font-weight: 700; font-size: 16px;">{source}</div>
            #                 <div style="color: white; font-size: 18px; font-weight: 600;">{value}</div>
            #             </div>
            #             """,
            #             unsafe_allow_html=True
            #         )
            # else:
            #     st.info("Score card data unavailable.")

            try:
                Ratings = omdb_data.get("Ratings", [])

                if Ratings:
                    # Create columns dynamically
                    cols = st.columns(len(Ratings))
                    for col, rating in zip(cols, Ratings):
                        source = rating.get("Source", "Unknown")
                        value = rating.get("Value", "N/A")
                        col.markdown(
                            f"""
                            <div style="
                                background: rgba(255,255,255,0.07);
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 15px;
                                text-align: center;
                                box-shadow: 0 0 15px rgba(0,0,0,0.3);
                            ">
                                <div style="color: #FFD700; font-weight: 700; font-size: 16px;">{source}</div>
                                <div style="color: white; font-size: 18px; font-weight: 600;">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Score card data unavailable.")

            except Exception as e:
                st.error(f"Error displaying ratings: {e}")


            st.markdown("---")

                        
            def get_video_url(movie_name):
                results = YoutubeSearch(movie_name + " official trailer", max_results=1).to_dict()
                if results:
                    video_id = results[0]['id']
                    return video_id,f"https://www.youtube.com/watch?v={video_id}"
                else:
                    return 0,"No video found."

            # video_id, video_url=get_video_url(title_input)
            try:
                video_id, video_url = get_video_url(title_input)
            except Exception as e:
                st.error(f"Failed to retrieve video URL: {e}")
                video_id, video_url = None, None

            # Replace with your API key
            API_KEY_YT = st.secrets["api_keys"]["API_KEY_YT"]

            # Initialize YouTube API client
            youtube = build('youtube', 'v3', developerKey=API_KEY_YT)

            def get_video_stats(video_id):
                # Get video statistics
                video_response = youtube.videos().list(
                    part='statistics,snippet',
                    id=video_id
                ).execute()

                if not video_response['items']:
                    return "Video not found."

                video_data = video_response['items'][0]
                stats = video_data['statistics']
                snippet = video_data['snippet']

                # Get channel statistics
                channel_id = snippet['channelId']
                channel_response = youtube.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()

                channel_stats = channel_response['items'][0]['statistics']

                return {
                    'title': snippet['title'],
                    'views': stats.get('viewCount', 'N/A'),
                    'likes': stats.get('likeCount', 'N/A'),
                    'comments': stats.get('commentCount', 'N/A'),
                    'subscribers': channel_stats.get('subscriberCount', 'N/A'),
                    'channel': snippet['channelTitle'],
                    'video_url': f"https://www.youtube.com/watch?v={video_id}"
                }

            # Example usage

            # video_data = get_video_stats(video_id)
            try:
                video_data = get_video_stats(video_id)
            except Exception as e:
                st.error(f"Error fetching video stats: {e}")
                video_data = None

            # --- Video Card ---
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.07);
                    padding: 20px;
                    border-radius: 15px;
                    color: #F5F5F5;
                    font-family: Arial, sans-serif;
                    background: rgba(255,255,255,0.07);
                    border: 1px solid rgba(255,215,0,0.25);
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 0 10px rgba(0,0,0,0.3);
                ">
                <a href="{video_data['video_url']}" target="_blank" style="
                        background: #FFD700;
                        color: #1F1F23;
                        font-weight: 700;
                        padding: 12px 25px;
                        border-radius: 8px;
                        text-decoration: none;
                        display: inline-block;
                        box-shadow: 0 0 8px rgba(0,0,0,0.3);
                        font-size: 16px;
                    ">
                        {video_data['title']}
                    </a>
                    <p style='margin:2px 0; font-size:14px;'>📺 Channel: <strong>{video_data['channel']}</strong> | 👥 Subscribers: <strong>{f"{int(video_data['subscribers']):,}"}</strong></p>
                    <div style="display:flex; gap:20px; margin-top:10px;">
                        <div style="text-align:center; flex:1; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,215,0,0.25); padding:10px; border-radius:12px;box-shadow: 0 0 15px rgba(0,0,0,0.3);">
                            👁️ <br><strong>{f"{int(video_data['views']):,}"}</strong> Views
                        </div>
                        <div style="text-align:center; flex:1; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,215,0,0.25); padding:10px; border-radius:12px;box-shadow: 0 0 15px rgba(0,0,0,0.3);">
                            👍 <br><strong>{f"{int(video_data['likes']):,}"}</strong> Likes
                        </div>
                        <div style="text-align:center; flex:1; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,215,0,0.25); padding:10px; border-radius:12px;box-shadow: 0 0 15px rgba(0,0,0,0.3);">
                            💬 <br><strong>{f"{int(video_data['comments']):,}"}</strong> Comments
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with tab2:

            ###################### YOU TUBE FUNCTIONALITY  #######################
            downloader = YoutubeCommentDownloader()
            comments_generator = downloader.get_comments_from_url(video_url)

            comments_data = []
            limit = 200
            count = 0

            for comment in comments_generator:
                comment_info = {
                    'text': comment.get('text', ''),
                    'author': comment.get('author', ''),
                    'time': comment.get('time', ''),
                    'likes': comment.get('likes', 0),
                    'replyCount': comment.get('replyCount', 0),
                    'channelId': comment.get('channelId', '')
                }
                comments_data.append(comment_info)
                count += 1
                if count >= limit:
                    break

            # print(comments_data)

            def format_comments_for_prompt(comments_data):
                formatted = ""
                for i, comment in enumerate(comments_data, 1):
                    text = comment.get('text', '').replace('\n', ' ')
                    author = comment.get('author', 'Unknown')
                    formatted += f"{i}. {text} (by {author})\n"
                return formatted

            comments_text = format_comments_for_prompt(comments_data)

            ### Feeding data to llm

            LLM_TOKEN_YOUTUBE = st.secrets["api_keys"]["LLM_TOKEN_YOUTUBE"]
            LLM_ENDPOINT = "https://models.github.ai/inference"
            MODEL_NAME = "openai/gpt-4.1"


            def get_movie_youtube_comments_summary(comments_text):
                prompt = f"""
                You are a movie critic and data analyst. Analyze the following YouTube comments for the movie: {comments_text}

                Given the youtube comments, generate the following metrics in JSON format:

                1. "Sentiment Analysis": 
                    a. "Public Opinion": provide one concise bullet point each for positive, negative, and neutral sentiment.
                    b. "Emotional Intensity": provide one concise bullet point each for key emotions such as love, disappointment, and anger.

                2. "Themes":
                    Provide 20 single-word topics with weights indicating their prominence in the movie comments.  
                    Example format → {{"Acting": 25, "Direction": 20, "Music": 15, "Plot": 18, "Complexity": 10}}

                3. "Audience Preferences": One bullet point each summarising comments on genre, cast, director
 
                4. "Expectations vs. Reality": One bullet point summarising viewers expectations based on the trailer and how well the movie met those expectations in two lines.
 
                5. "Memorable Quotes": Identify the most talked-about scenes from the comments and describe in two lines.
                
                6. "Criticism": Spot specific viewer complaints—like pacing issues, plot holes, or weak performances—to understand what didn’t resonate in the trailer (in bullet points)
        
                7. "Viewer engagement": Does viewer mention wanting to see sequels or next season (use bullet points)
 
                8. "Cultural Insights":
                    "Cultural References & Values": Cultural references or values that comments identify with or react against (in one bullet point)
                    "Social Issues & Generational Perspectives": How the comments reflect or engage with social issues, identity, or generational perspectives (in 3 bullet points)
                    "Emotional Tone & Viewer Mindset": The emotional tone and viewer mindset, including specific triggers that evoke strong reactions—such as excitement, nostalgia, discomfort, or curiosity (in 3 bullet points)
                        
                9. "Production Review": Comments on visuals, sound, effects, or direction quality. Provide exactly 3 concise bullet points
                
                10. "Narrative Structure & Plot Complexity": Remarks on pacing, clarity, twists, or ending preferences. Provide exactly 3 concise bullet points
                
                11. "Aesthetics": Opinions on costume, set design, color, or overall aesthetics. Provide exactly 3 concise bullet points
        
                Provide output strictly as valid JSON only with these keys exactly.
                """

                messages = [
                    {"role": "system", "content": "You are a movie critic and data analyst."},
                    {"role": "user", "content": prompt}
                ]

                client = ChatCompletionsClient(
                    endpoint=LLM_ENDPOINT,
                    credential=AzureKeyCredential(LLM_TOKEN_YOUTUBE)
                )

                response = client.complete(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=20000
                )
                
                content = response.choices[0].message.content.strip()
                # st.write("LLM raw output:", content)
                metrics = json.loads(content)
                return metrics
            
                ###################### REDDIT FUNCTIONALITY  ######################################################################################

            def fetch_reddit_posts(title_input, limit=15):
                url = f"https://www.reddit.com/search.json?q={title_input}&limit={limit}"
                headers = {'User-agent': 'MovieBot 0.1'}
                response = requests.get(url, headers=headers)
                
                try:
                    data = response.json()
                    posts_data = [
                        {
                            "title": post["data"].get("title", ""),
                            "selftext": post["data"].get("selftext", ""),
                            "author": post["data"].get("author", ""),
                            "subreddit": post["data"].get("subreddit", ""),
                            "permalink": f"https://www.reddit.com{post['data'].get('permalink', '')}",
                            "score": post["data"].get("score", 0),
                            "num_comments": post["data"].get("num_comments", 0)
                        }
                        for post in data["data"]["children"]
                    ]
                    
                    return posts_data
                
                except ValueError:
                    return []

            def format_posts_for_prompt(posts_data):
                formatted = ""
                for i, post in enumerate(posts_data, 1):
                    title = post.get('title', '').replace('\n', ' ').strip()
                    selftext = post.get('selftext', '').replace('\n', ' ').strip()
                    author = post.get('author', 'Unknown')
                    subreddit = post.get('subreddit', 'Unknown')
                    permalink = post.get('permalink', '')
                    score = post.get('score', 0)
                    num_comments = post.get('num_comments', 0)

                    content = title
                    if selftext:
                        content += " - " + selftext

                    formatted += (
                        f"{i}. {content} "
                        f"(Author: {author}, Subreddit: {subreddit}, Score: {score}, Comments: {num_comments})\n"
                        f"Link: {permalink}\n\n"
                    )
                return formatted
            
            # posts = fetch_reddit_posts(title_input)
            # formatted_text = format_posts_for_prompt(posts)
            try:
                posts = fetch_reddit_posts(title_input)
                formatted_text = format_posts_for_prompt(posts)
            except Exception as e:
                st.error(f"Error: {e}")

            ### Feeding data to llm
            LLM_TOKEN_REDDIT = st.secrets["api_keys"]["LLM_TOKEN_REDDIT"]
            LLM_ENDPOINT = "https://models.github.ai/inference"
            MODEL_NAME = "openai/gpt-4.1"

            def get_movie_reddit_posts_summary(formatted_text):
                prompt = f"""
                You are a movie critic and data analyst. Analyze the following reddit posts for the movie: {formatted_text}

                Given the youtube comments, generate the following metrics in JSON format:

                1. "Sentiment Analysis": 
                    a. "Public Opinion": provide one concise bullet point each for positive, negative, and neutral sentiment.
                    b. "Emotional Intensity": provide one concise bullet point each for key emotions such as love, disappointment, and anger.
                2. "Themes":
                    Provide 20 single-word topics with weights indicating their prominence in the movie comments.  
                    Example format → {{"Acting": 25, "Direction": 20, "Music": 15, "Plot": 18, "Complexity": 10}}
                3. "Audience Preferences": One bullet point each summarising comments on genre, cast, director
                4. "Expectations vs. Reality": One bullet point summarising viewers expectations based on the trailer and how well the movie met those expectations in two lines.
                5. "Memorable Quotes": Identify the most talked-about scenes from the comments and describe in two lines.        
                6. "Criticism": Spot specific viewer complaints—like pacing issues, plot holes, or weak performances—to understand what didn’t resonate in the trailer (in bullet points)
                7. "Viewer engagement": Does viewer mention wanting to see sequels or next season (use bullet points)
                8. "Cultural Insights":
                    "Cultural References & Values": Cultural references or values that comments identify with or react against (in one bullet point)
                    "Social Issues & Generational Perspectives": How the comments reflect or engage with social issues, identity, or generational perspectives (in 3 bullet points)
                    "Emotional Tone & Viewer Mindset": The emotional tone and viewer mindset, including specific triggers that evoke strong reactions—such as excitement, nostalgia, discomfort, or curiosity (in 3 bullet points)       
                9. "Production Review": Comments on visuals, sound, effects, or direction quality. Provide exactly 3 concise bullet points
                10. "Narrative Structure & Plot Complexity": Remarks on pacing, clarity, twists, or ending preferences. Provide exactly 3 concise bullet points
                11. "Aesthetics": Opinions on costume, set design, color, or overall aesthetics. Provide exactly 3 concise bullet points
                Provide output strictly as valid JSON only with these keys exactly.
                """
                messages = [
                    {"role": "system", "content": "You are a movie critic and data analyst."},
                    {"role": "user", "content": prompt}
                ]

                client = ChatCompletionsClient(
                    endpoint=LLM_ENDPOINT,
                    credential=AzureKeyCredential(LLM_TOKEN_REDDIT)
                )

                response = client.complete(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=20000
                ) 
                content_red = response.choices[0].message.content.strip()
                #st.write("LLM raw output REDDIT:", content_red)
                metrics_red = json.loads(content_red)
                return metrics_red

            ###################### GOOGLE PLAY MOVIES FUNCTIONALITY  #########################

            ### Feeding data to llm
            LLM_TOKEN_GOOGLE = st.secrets["api_keys"]["LLM_TOKEN_GOOGLE"]
            LLM_ENDPOINT = "https://models.github.ai/inference"
            MODEL_NAME = "openai/gpt-4.1"

            def get_movie_google_play_reviews_summary(title_input: str) -> str:
                prompt = f""" Please provide the following data for [specific topic or entity], including the source of the information for each data point
                You are a movie analyst. For the movie '{title_input}', provide:

                Please provide:
                1. "Sentiment Analysis": 
                    a. "Public Opinion": provide one concise bullet point each for positive, negative, and neutral sentiment.
                    b. "Emotional Intensity": provide one concise bullet point each for key emotions such as love, disappointment, and anger.
                2. "Themes":
                    Provide 20 single-word topics with weights indicating their prominence in the movie comments.  
                    Example format → {{"Acting": 25, "Direction": 20, "Music": 15, "Plot": 18, "Complexity": 10}}
                3. "Audience Preferences": One bullet point each summarising comments on genre, cast, director
                4. "Expectations vs. Reality": One bullet point summarising viewers expectations based on the trailer and how well the movie met those expectations in two lines.
                5. "Memorable Quotes": Identify the most talked-about scenes from the comments and describe in two lines.        
                6. "Criticism": Spot specific viewer complaints—like pacing issues, plot holes, or weak performances—to understand what didn’t resonate in the trailer (in bullet points)
                7. "Viewer engagement": Does viewer mention wanting to see sequels or next season (use bullet points)
                8. "Cultural Insights":
                    "Cultural References & Values": Cultural references or values that comments identify with or react against (in one bullet point)
                    "Social Issues & Generational Perspectives": How the comments reflect or engage with social issues, identity, or generational perspectives (in 3 bullet points)
                    "Emotional Tone & Viewer Mindset": The emotional tone and viewer mindset, including specific triggers that evoke strong reactions—such as excitement, nostalgia, discomfort, or curiosity (in 3 bullet points)       
                9. "Production Review": Comments on visuals, sound, effects, or direction quality. Provide exactly 3 concise bullet points
                10. "Narrative Structure & Plot Complexity": Remarks on pacing, clarity, twists, or ending preferences. Provide exactly 3 concise bullet points
                11. "Aesthetics": Opinions on costume, set design, color, or overall aesthetics. Provide exactly 3 concise bullet points

                Note: Refer only google play movies data no other data source is accepted
                Provide output strictly as valid JSON only with these keys exactly.
                """
                messages = [
                    {"role": "system", "content": "You are a movie critic and data analyst."},
                    {"role": "user", "content": prompt}
                ]

                client = ChatCompletionsClient(
                    endpoint=LLM_ENDPOINT,
                    credential=AzureKeyCredential(LLM_TOKEN_GOOGLE)
                )

                response = client.complete(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=20000
                ) 
                content_google = response.choices[0].message.content.strip()
                #st.write("LLM raw output GOOGLE:", content_google)
                metrics_google = json.loads(content_google)
                return metrics_google


            ###################### END FUNCTIONALITY  ##################################################################

            st.markdown(
            """
            <h3 style='color: #FFD700; text-align:center; font-weight: bold; margin-top:30px;'>📊 Audience Insights & Analysis</h3>
            """,
            unsafe_allow_html=True
        )
            if comments_text is None:
                st.error("comments not found in youtube. Please try another title.")
            
            else:
                # Fetch LLM metrics live
                st.markdown("""
                <style>
                .stSpinner p { color: white !important; }
                </style>
                """, unsafe_allow_html=True)
                with st.spinner("Fetching Social Media Insights from LLM..."):

                    #Exception Handling for LLM calls
                    try:
                        llm_metrics = get_movie_youtube_comments_summary(comments_text)
                    except Exception as e:
                        st.error(f"Failed to get YouTube comments summary: {e}")
                        llm_metrics = None

                    try:
                        llm_metrics_red = get_movie_reddit_posts_summary(formatted_text)
                    except Exception as e:
                        st.error(f"Failed to get Reddit posts summary: {e}")
                        llm_metrics_red = None

                    try:
                        llm_metrics_google = get_movie_google_play_reviews_summary(title_input.strip())
                    except Exception as e:
                        st.error(f"Failed to get Google Play reviews summary: {e}")
                        llm_metrics_google = None


                tabs = st.tabs(["Youtube", "Reddit", "Google"])

                with tabs[0]:
                    if llm_metrics:
                        def display_card(title, content):
                            st.markdown(f"""
                            <div style="
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 15px;
                                margin-bottom: 15px;
                                background: rgba(255,255,255,0.07);
                                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                                <div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">
                                    {title}
                                </div>
                                <p style="
                                    color: white;
                                    font-size: 15px;
                                    line-height: 1.5;
                                    margin: 0;
                                ">
                                    {content}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        
                        def display_titles(title):
                            st.markdown(
                            f"""
                            <div style="
                                background: rgba(255,255,255,0.07);
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 10px 0;
                                text-align: center;
                                color: #FFD700;
                                font-weight: 700;
                                font-size: 18px;
                                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                                {title}
                            </div>
                            <div style="height: 10px;"></div>  <!-- Small spacer -->
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Utility function to display lists as HTML bullets
                        def display_bullets(title, points):

                            if points:
                                formatted = "<br>".join([f"• {p}" for p in points])
                                display_card(title, formatted)
                            else:
                                display_card(title, "No data available.")
                                    
                        #--------Sentiment Analysis-------#

                        sentiment_data = llm_metrics.get("Sentiment Analysis", "N/A")

                        if sentiment_data == "N/A" or not sentiment_data:
                            st.info("No sentiment data available.")
                        else:
                            display_titles("🎭 Sentiment Analysis")
        
                            # Public Opinion
                            overall = sentiment_data.get("Public Opinion", {})
                            
                            if overall:
                            
                                st.markdown("""<div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">📊 Public Opinion</div>
                                    """, unsafe_allow_html=True)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    display_card("😍 Positive", overall.get("Positive", "N/A"))

                                with col2:
                                    display_card("😤 Negative", overall.get("Negative", "N/A"))

                                with col3:
                                    display_card("😐 Neutral", overall.get("Neutral", "N/A"))

                            else:
                                display_card("📊 Public Opinion", "No data available.")

                            # Emotional Intensity
                            emotional = sentiment_data.get("Emotional Intensity", {})
    
                            if emotional:
            
                                st.markdown("""<div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">💫 Emotional Intensity</div>
                                    """, unsafe_allow_html=True)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    display_card("💖 Love", emotional.get("Love", "N/A"))

                                with col2:
                                    display_card("💔 Disappointment", emotional.get("Disappointment", "N/A"))

                                with col3:
                                    display_card("😡 Anger", emotional.get("Anger", "N/A"))

                            else:
                                display_card("💫 Emotional Intensity", "No data available.")
                            
                            display_titles("📚 Themes")

                        topics = llm_metrics.get("Themes", {}) or {}

                        if topics:
                            # Prepare weights
                            topic_weights = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

                            # Reduce size for compact display
                            wc = WordCloud(
                                width=800,
                                height=400,
                                background_color="#f9f9f9",
                                colormap="viridis_r",
                                prefer_horizontal=0.9,
                                max_words=50,
                                max_font_size=80,
                                min_font_size=10,
                                normalize_plurals=True,
                                random_state=42,
                                scale=3
                            ).generate_from_frequencies(topic_weights)

                            # Display
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.imshow(wc, interpolation="antialiased")
                            ax.axis("off")
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        else:
                            st.info("No Themes found for this movie.")



                        audience_data = llm_metrics.get("Audience Preferences", {})

                        if audience_data:
                            display_titles("💬 Audience Preferences")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                display_card("📖 Genre", audience_data.get("Genre", "N/A"))

                            with col2:
                                display_card("🌟 Cast", audience_data.get("Cast", "N/A"))

                            with col3:
                                display_card("🎥 Director", audience_data.get("Director", "N/A"))

                        else:
                            display_card("💬 Audience Preferences", "No data available.")
                        
                        
                        criticism = llm_metrics.get("Criticism", [])

                        viewer_engagement = llm_metrics.get("Viewer engagement", [])

                        cultural_insights = llm_metrics.get("Cultural Insights", {})
    
                        production_review = llm_metrics.get("Production Review", [])

                        narrative_structure = llm_metrics.get("Narrative Structure & Plot Complexity", [])

                        aesthetics = llm_metrics.get("Aesthetics", [])

                        # Criticism
                        display_bullets("🛑 Criticism", criticism)

                        # Viewer Engagement
                        display_bullets("👥 Viewer Engagement", viewer_engagement)

                        col1,col2=st.columns(2)
                        with col1:
                            display_card("⚖️ Expectations vs. Reality", llm_metrics.get("Expectations vs. Reality","N/A"))
                        with col2:
                            display_card("🌟 Memorable Quotes", llm_metrics.get("Memorable Quotes","N/A"))
                            
                        cultural_insights = llm_metrics.get("Cultural Insights", {})

                        if cultural_insights:
                            formatted_list = []

                            for key, value in cultural_insights.items():
                                if value:  
                                    if isinstance(value, list):
                                        formatted_list.append(f"<br><strong>{key}</strong>")
                                        formatted_list.extend([f"&emsp;&emsp;• {p}" for p in value if p])
                                    else:
                                        formatted_list.append(f"<strong>{key}</strong>:<br>&emsp;&emsp;{value}")

                            if formatted_list:
                                formatted = "<br>".join(formatted_list)
                                display_card("🌏 Cultural Insights", formatted)
                            else:
                                display_card("🌏 Cultural Insights", "No data available.")
                        else:
                            display_card("🌏 Cultural Insights", "No data available.")


                        # Production Review
                        display_bullets("🎬 Production Review", production_review)

                        # Narrative Structure & Plot Complexity
                        display_bullets("📖 Narrative Structure & Plot Complexity", narrative_structure)

                        # Aesthetics
                        display_bullets("🎨 Aesthetics", aesthetics)

                with tabs[1]:
                    if llm_metrics_red:
                        def display_card(title, content):
                            st.markdown(f"""
                            <div style="
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 15px;
                                margin-bottom: 15px;
                                background: rgba(255,255,255,0.07);
                                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                                <div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">
                                    {title}
                                </div>
                                <p style="
                                    color: white;
                                    font-size: 15px;
                                    line-height: 1.5;
                                    margin: 0;
                                ">
                                    {content}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        
                        def display_titles(title):
                            st.markdown(
                            f"""
                            <div style="
                                background: rgba(255,255,255,0.07);
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 10px 0;
                                text-align: center;
                                color: #FFD700;
                                font-weight: 700;
                                font-size: 18px;
                                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                                {title}
                            </div>
                            <div style="height: 10px;"></div>  <!-- Small spacer -->
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Utility function to display lists as HTML bullets
                        def display_bullets(title, points):

                            if points:
                                formatted = "<br>".join([f"• {p}" for p in points])
                                display_card(title, formatted)
                            else:
                                display_card(title, "No data available.")
                                    
                        #--------Sentiment Analysis-------#

                        sentiment_data = llm_metrics_red.get("Sentiment Analysis", "N/A")

                        if sentiment_data == "N/A" or not sentiment_data:
                            st.info("No sentiment data available.")
                        else:
                            display_titles("🎭 Sentiment Analysis")
        
                            # Public Opinion
                            overall = sentiment_data.get("Public Opinion", {})
                            
                            if overall:
                            
                                st.markdown("""<div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">📊 Public Opinion</div>
                                    """, unsafe_allow_html=True)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    display_card("😍 Positive", overall.get("Positive", "N/A"))

                                with col2:
                                    display_card("😤 Negative", overall.get("Negative", "N/A"))

                                with col3:
                                    display_card("😐 Neutral", overall.get("Neutral", "N/A"))

                            else:
                                display_card("📊 Public Opinion", "No data available.")

                            # Emotional Intensity
                            emotional = sentiment_data.get("Emotional Intensity", {})
    
                            if emotional:
            
                                st.markdown("""<div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">💫 Emotional Intensity</div>
                                    """, unsafe_allow_html=True)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    display_card("💖 Love", emotional.get("Love", "N/A"))

                                with col2:
                                    display_card("💔 Disappointment", emotional.get("Disappointment", "N/A"))

                                with col3:
                                    display_card("😡 Anger", emotional.get("Anger", "N/A"))

                            else:
                                display_card("💫 Emotional Intensity", "No data available.")
                            
                            display_titles("📚 Themes")

                        topics = llm_metrics_red.get("Themes", {}) or {}

                        if topics:
                            # Prepare weights
                            topic_weights = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

                            # Reduce size for compact display
                            wc = WordCloud(
                                width=800,
                                height=400,
                                background_color="#f9f9f9",
                                colormap="viridis_r",
                                prefer_horizontal=0.9,
                                max_words=50,
                                max_font_size=80,
                                min_font_size=10,
                                normalize_plurals=True,
                                random_state=42,
                                scale=3
                            ).generate_from_frequencies(topic_weights)

                            # Display
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.imshow(wc, interpolation="antialiased")
                            ax.axis("off")
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        else:
                            st.info("No Themes found for this movie.")



                        audience_data = llm_metrics_red.get("Audience Preferences", {})

                        if audience_data:
                            display_titles("💬 Audience Preferences")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                display_card("📖 Genre", audience_data.get("Genre", "N/A"))

                            with col2:
                                display_card("🌟 Cast", audience_data.get("Cast", "N/A"))

                            with col3:
                                display_card("🎥 Director", audience_data.get("Director", "N/A"))

                        else:
                            display_card("💬 Audience Preferences", "No data available.")
                        
                        
                        criticism = llm_metrics_red.get("Criticism", [])

                        viewer_engagement = llm_metrics_red.get("Viewer engagement", [])

                        cultural_insights = llm_metrics_red.get("Cultural Insights", {})
    
                        production_review = llm_metrics_red.get("Production Review", [])

                        narrative_structure = llm_metrics_red.get("Narrative Structure & Plot Complexity", [])

                        aesthetics = llm_metrics_red.get("Aesthetics", [])

                        # Criticism
                        display_bullets("🛑 Criticism", criticism)

                        # Viewer Engagement
                        display_bullets("👥 Viewer Engagement", viewer_engagement)

                        col1,col2=st.columns(2)
                        with col1:
                            display_card("⚖️ Expectations vs. Reality", llm_metrics_red.get("Expectations vs. Reality","N/A"))
                        with col2:
                            display_card("🌟 Memorable Quotes", llm_metrics_red.get("Memorable Quotes","N/A"))
                            
                        cultural_insights = llm_metrics_red.get("Cultural Insights", {})

                        if cultural_insights:
                            formatted_list = []

                            for key, value in cultural_insights.items():
                                if value:  
                                    if isinstance(value, list):
                                        formatted_list.append(f"<br><strong>{key}</strong>")
                                        formatted_list.extend([f"&emsp;&emsp;• {p}" for p in value if p])
                                    else:
                                        formatted_list.append(f"<strong>{key}</strong>:<br>&emsp;&emsp;{value}")

                            if formatted_list:
                                formatted = "<br>".join(formatted_list)
                                display_card("🌏 Cultural Insights", formatted)
                            else:
                                display_card("🌏 Cultural Insights", "No data available.")
                        else:
                            display_card("🌏 Cultural Insights", "No data available.")


                        # Production Review
                        display_bullets("🎬 Production Review", production_review)

                        # Narrative Structure & Plot Complexity
                        display_bullets("📖 Narrative Structure & Plot Complexity", narrative_structure)

                        # Aesthetics
                        display_bullets("🎨 Aesthetics", aesthetics)

                with tabs[2]:
                    if llm_metrics_google:
                        def display_card(title, content):
                            st.markdown(f"""
                            <div style="
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 15px;
                                margin-bottom: 15px;
                                background: rgba(255,255,255,0.07);
                                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                                <div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">
                                    {title}
                                </div>
                                <p style="
                                    color: white;
                                    font-size: 15px;
                                    line-height: 1.5;
                                    margin: 0;
                                ">
                                    {content}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        
                        def display_titles(title):
                            st.markdown(
                            f"""
                            <div style="
                                background: rgba(255,255,255,0.07);
                                border: 1px solid rgba(255,215,0,0.25);
                                border-radius: 12px;
                                padding: 10px 0;
                                text-align: center;
                                color: #FFD700;
                                font-weight: 700;
                                font-size: 18px;
                                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                            ">
                                {title}
                            </div>
                            <div style="height: 10px;"></div>  <!-- Small spacer -->
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Utility function to display lists as HTML bullets
                        def display_bullets(title, points):

                            if points:
                                formatted = "<br>".join([f"• {p}" for p in points])
                                display_card(title, formatted)
                            else:
                                display_card(title, "No data available.")
                                    
                        #--------Sentiment Analysis-------#

                        sentiment_data = llm_metrics_google.get("Sentiment Analysis", "N/A")

                        if sentiment_data == "N/A" or not sentiment_data:
                            st.info("No sentiment data available.")
                        else:
                            display_titles("🎭 Sentiment Analysis")
        
                            # Public Opinion
                            overall = sentiment_data.get("Public Opinion", {})
                            
                            if overall:
                            
                                st.markdown("""<div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">📊 Public Opinion</div>
                                    """, unsafe_allow_html=True)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    display_card("😍 Positive", overall.get("Positive", "N/A"))

                                with col2:
                                    display_card("😤 Negative", overall.get("Negative", "N/A"))

                                with col3:
                                    display_card("😐 Neutral", overall.get("Neutral", "N/A"))

                            else:
                                display_card("📊 Public Opinion", "No data available.")

                            # Emotional Intensity
                            emotional = sentiment_data.get("Emotional Intensity", {})
    
                            if emotional:
            
                                st.markdown("""<div style="
                                    text-align: center;
                                    color: #FFD700;
                                    font-weight: 700;
                                    font-size: 18px;
                                    margin-bottom: 10px;
                                ">💫 Emotional Intensity</div>
                                    """, unsafe_allow_html=True)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    display_card("💖 Love", emotional.get("Love", "N/A"))

                                with col2:
                                    display_card("💔 Disappointment", emotional.get("Disappointment", "N/A"))

                                with col3:
                                    display_card("😡 Anger", emotional.get("Anger", "N/A"))

                            else:
                                display_card("💫 Emotional Intensity", "No data available.")
                            
                            display_titles("📚 Themes")

                        topics = llm_metrics_google.get("Themes", {}) or {}

                        if topics:
                            # Prepare weights
                            topic_weights = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

                            # Reduce size for compact display
                            wc = WordCloud(
                                width=800,
                                height=400,
                                background_color="#f9f9f9",
                                colormap="viridis_r",
                                prefer_horizontal=0.9,
                                max_words=50,
                                max_font_size=80,
                                min_font_size=10,
                                normalize_plurals=True,
                                random_state=42,
                                scale=3
                            ).generate_from_frequencies(topic_weights)

                            # Display
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.imshow(wc, interpolation="antialiased")
                            ax.axis("off")
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        else:
                            st.info("No Themes found for this movie.")

                        audience_data = llm_metrics_google.get("Audience Preferences", {})

                        if audience_data:
                            display_titles("💬 Audience Preferences")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                display_card("📖 Genre", audience_data.get("Genre", "N/A"))

                            with col2:
                                display_card("🌟 Cast", audience_data.get("Cast", "N/A"))

                            with col3:
                                display_card("🎥 Director", audience_data.get("Director", "N/A"))

                        else:
                            display_card("💬 Audience Preferences", "No data available.")
                        
                        
                        criticism = llm_metrics_google.get("Criticism", [])

                        viewer_engagement = llm_metrics_google.get("Viewer engagement", [])

                        cultural_insights = llm_metrics_google.get("Cultural Insights", {})
    
                        production_review = llm_metrics_google.get("Production Review", [])

                        narrative_structure = llm_metrics_google.get("Narrative Structure & Plot Complexity", [])

                        aesthetics = llm_metrics_google.get("Aesthetics", [])

                        # Criticism
                        display_bullets("🛑 Criticism", criticism)

                        # Viewer Engagement
                        display_bullets("👥 Viewer Engagement", viewer_engagement)

                        col1,col2=st.columns(2)
                        with col1:
                            display_card("⚖️ Expectations vs. Reality", llm_metrics_google.get("Expectations vs. Reality","N/A"))
                        with col2:
                            display_card("🌟 Memorable Quotes", llm_metrics_google.get("Memorable Quotes","N/A"))
                            
                        cultural_insights = llm_metrics_google.get("Cultural Insights", {})

                        if cultural_insights:
                            formatted_list = []

                            for key, value in cultural_insights.items():
                                if value:  
                                    if isinstance(value, list):
                                        formatted_list.append(f"<br><strong>{key}</strong>")
                                        formatted_list.extend([f"&emsp;&emsp;• {p}" for p in value if p])
                                    else:
                                        formatted_list.append(f"<strong>{key}</strong>:<br>&emsp;&emsp;{value}")

                            if formatted_list:
                                formatted = "<br>".join(formatted_list)
                                display_card("🌏 Cultural Insights", formatted)
                            else:
                                display_card("🌏 Cultural Insights", "No data available.")
                        else:
                            display_card("🌏 Cultural Insights", "No data available.")


                        # Production Review
                        display_bullets("🎬 Production Review", production_review)

                        # Narrative Structure & Plot Complexity
                        display_bullets("📖 Narrative Structure & Plot Complexity", narrative_structure)

                        # Aesthetics
                        display_bullets("🎨 Aesthetics", aesthetics)
