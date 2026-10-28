import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import re

# NLTK 리소스 다운로드 (필요 시)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

st.set_page_config(
    page_title="유튜브 댓글 종합 분석기",
    page_icon="📊",
    layout="wide"
)

# --- YouTube API 클라이언트 생성 함수 ---
@st.cache_resource
def get_youtube_client(api_key):
    return build('youtube', 'v3', developerKey=api_key)

# --- 댓글 수집 함수 ---
def fetch_youtube_comments(youtube, video_id, max_comments=100):
    comments_data = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            textFormat="plainText"
        )
        response = request.execute()
        
        while response and len(comments_data) < max_comments:
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'author': snippet['authorDisplayName'],
                    'text': snippet['textDisplay'],
                    'likes': snippet['likeCount'],
                    'published_at': snippet['publishedAt']
                })
            
            # 다음 페이지가 있으면 계속 가져오기
            if 'nextPageToken' in response and len(comments_data) < max_comments:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(max_comments - len(comments_data), 100),
                    pageToken=response['nextPageToken'],
                    textFormat="plainText"
                )
                response = request.execute()
            else:
                break
    except HttpError as e:
        st.error(f"YouTube API 에러 발생: {e}")
        return []
    except Exception as e:
        st.error(f"댓글을 불러오는 중 오류가 발생했습니다: {e}")
        return []
        
    return comments_data

# --- URL에서 Video ID 추출 함수 ---
def extract_video_id(url_or_id):
    if len(url_or_id) == 11 and not ("/" in url_or_id or "." in url_or_id):
        return url_or_id
    
    # youtu.be/ID 형식
    match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url_or_id)
    if match:
        return match.group(1)
    
    # watch?v=ID 형식
    match = re.search(r'v=([a-zA-Z0-9_-]{11})', url_or_id)
    if match:
        return match.group(1)
        
    return url_or_id

# --- UI 레이아웃 ---
st.title("📊 유튜브 댓글 종합 분석 대시보드")
st.markdown("유튜브 영상 URL을 입력하면 **최다 좋아요 댓글, 감정 분석, 워드클라우드, 주요 키워드** 등을 한눈에 분석해 줍니다.")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    # Streamlit secrets 또는 직접 입력 지원
    api_key_input = st.secrets.get("YOUTUBE_API_KEY", "")
    api_key = st.text_input("YouTube Data API Key", value=api_key_input, type="password")
    
    max_comments = st.slider("수집할 최대 댓글 수", min_value=50, max_value=500, value=100, step=50)
    st.info("💡 팁: API 키는 Streamlit Cloud Secrets에 등록해두면 안전합니다.")

# 메인 입력창
video_input = st.text_input("유튜브 영상 링크 또는 Video ID를 입력하세요", placeholder="https://www.youtube.com/watch?v=xxxxxxxxx")

if st.button("분석 시작하기", type="primary"):
    if not api_key:
        st.warning("⚠️ YouTube API Key를 입력해주세요.")
    elif not video_input:
        st.warning("⚠️ 유튜브 영상 링크나 ID를 입력해주세요.")
    else:
        video_id = extract_video_id(video_input)
        
        with st.spinner("🔄 댓글을 수집하고 분석하는 중입니다... 잠시만 기다려주세요!"):
            youtube = get_youtube_client(api_key)
            raw_comments = fetch_youtube_comments(youtube, video_id, max_comments)
            
            if not raw_comments:
                st.error("❌ 수집된 댓글이 없거나 올바르지 않은 영상 ID입니다. (댓글이 비어있거나 막힌 영상일 수 있습니다.)")
            else:
                df = pd.DataFrame(raw_comments)
                
                # 데이터 전처리 및 감정 분석 추가
                sentiments = []
                for text in df['text']:
                    # TextBlob을 활용한 단순 감정 분석 (Polarity: -1 ~ 1)
                    blob = TextBlob(str(text))
                    polarity = blob.sentiment.polarity
                    if polarity > 0.1:
                        sentiments.append('긍정 😊')
                    elif polarity < -0.1:
                        sentiments.append('부정 😢')
                    else:
                        sentiments.append('중립 😐')
                df['sentiment'] = sentiments

                st.success(f"🎉 총 {len(df)}개의 댓글을 성공적으로 분석했습니다!")
                
                # --- 섹션 1: 핵심 지표 요약 ---
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("수집된 댓글 수", f"{len(df):,}개")
                col2.metric("총 좋아요 수 합계", f"{df['likes'].sum():,}개")
                col3.metric("평균 좋아요 수", f"{df['likes'].mean():.1f개")

                # --- 섹션 2: 최다 좋아요 댓글 TOP 3 ---
                st.subheader("🏆 좋아요를 가장 많이 받은 댓글 TOP 3")
                top_liked = df.sort_values(by='likes', ascending=False).head(3)
                for idx, row in top_liked.reset_index(drop=True).iterrows():
                    st.markdown(f"""
                    > **{row['author']}** (👍 좋아요: `{row['likes']:,}개`)  
                    > {row['text']}
                    """)

                # --- 섹션 3: 감정 분석 결과 ---
                st.divider()
                st.subheader("📈 댓글 감정 분석 비율")
                sentiment_counts = df['sentiment'].value_counts()
                
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.dataframe(sentiment_counts, use_container_width=True)
                with col_chart2:
                    fig, ax = plt.subplots()
                    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=['#66b3ff','#99ff99','#ff9999'])
                    ax.set_ylabel('')
                    st.pyplot(fig)

                # --- 섹션 4: 워드클라우드 & 키워드 분석 ---
                st.divider()
                st.subheader("☁️ 댓글 워드클라우드 & 주요 키워드")
                
                all_text = " ".join(df['text'].astype(str).tolist())
                if len(all_text.strip()) > 0:
                    try:
                        # 워드클라우드 생성 (한글 폰트 문제 방지를 위해 기본 폰트나 시스템 폰트 적용 필요 시 설정)
                        wordcloud = WordCloud(
                            width=800, height=400, 
                            background_color='white',
                            max_words=100
                        ).generate(all_text)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.info("워드클라우드를 생성할 수 있는 텍스트 데이터가 부족합니다.")
                
                # --- 섹션 5: 전체 댓글 데이터 테이블 ---
                st.divider()
                st.subheader("📋 전체 수집 댓글 데이터 보기")
                st.dataframe(df[['author', 'text', 'likes', 'sentiment', 'published_at']], use_container_width=True)
                
                # CSV 다운로드 버튼
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 분석 결과 CSV 다운로드",
                    data=csv,
                    file_name="youtube_comments_analysis.csv",
                    mime="text/csv",
                )

---

### 💡 GitHub 배포 및 API 키 설정 팁
1. **GitHub 업로드:** 
   * 위 2개의 파일(`app.py`, `requirements.txt`)을 본인의 GitHub 리포지토리에 커밋하고 푸시합니다.
2. **Streamlit Cloud 연동:**
   * [Streamlit Cloud](https://share.streamlit.io/)에 접속하여 가입 후 **New app**을 클릭합니다.
   * 방금 생성한 GitHub 리포지토리와 브랜치, Main file path(`app.py`)를 지정합니다.
3. **API Key 보안 설정 (선택):**
   * 앱 설정(Advanced settings) -> **Secrets** 탭에 아래와 같이 입력해두면 코드를 고치지 않고 안전하게 API 키를 연동할 수 있습니다.
     ```toml
     YOUTUBE_API_KEY = "본인의_유튜브_API_키"
     ```

---
<FollowUp>
[
  "YouTube Data API 키 발급받는 방법을 알려줘",
  "이 코드에 한국어 형태소 분석기(KoNLPy)를 연동해서 키워드 정밀도를 높이는 법을 알려줘",
  "Render나 다른 플랫폼에 배포할 때의 차이점을 설명해줘"
]
</FollowUp>
