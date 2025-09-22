import os
import io
import re
from typing import Optional, List, Tuple, Dict

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text

# VADER sentiment analyzer (lightweight)
_vader = None


APP_TITLE = "고객 피드백 분석"
DEFAULT_FILE_NAME = "@feedback-data.csv"


def find_default_csv_file() -> Optional[str]:
    workspace_dir = os.getcwd()
    candidate_path = os.path.join(workspace_dir, DEFAULT_FILE_NAME)
    if os.path.exists(candidate_path):
        return candidate_path
    return None


@st.cache_data(show_spinner=False, ttl=300)  # 5분 캐시
def load_dataframe(file_like_or_path) -> pd.DataFrame:
    if isinstance(file_like_or_path, (str, os.PathLike)):
        return pd.read_csv(file_like_or_path)
    else:
        file_like_or_path.seek(0)
        return pd.read_csv(file_like_or_path)


def ensure_text_column_ui(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    st.subheader("데이터 컬럼 매핑")
    cols = list(df.columns)
    # 추정: 텍스트로 보이는 컬럼 우선 선택
    default_text_col = None
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["text", "내용", "피드백", "comment", "리뷰", "review"]):
            default_text_col = c
            break
    text_col = st.selectbox("텍스트 컬럼", options=cols, index=(cols.index(default_text_col) if default_text_col in cols else 0))

    # 날짜/카테고리(제품군) 컬럼 선택은 선택 사항
    date_col = st.selectbox("날짜 컬럼(선택)", options=["(없음)"] + cols)
    category_col = st.selectbox("카테고리/제품군 컬럼(선택)", options=["(없음)"] + cols)

    date_col = None if date_col == "(없음)" else date_col
    category_col = None if category_col == "(없음)" else category_col
    return text_col, date_col, category_col


def apply_filters(df: pd.DataFrame, date_col: Optional[str], category_col: Optional[str]) -> pd.DataFrame:
    filtered = df.copy()
    if date_col and date_col in filtered.columns:
        with st.expander("기간 필터"):
            # 날짜 파싱 시도
            parsed = pd.to_datetime(filtered[date_col], errors="coerce")
            filtered = filtered.assign(_parsed_date=parsed)
            min_d = filtered["_parsed_date"].min()
            max_d = filtered["_parsed_date"].max()
            if pd.notna(min_d) and pd.notna(max_d):
                start, end = st.date_input("날짜 범위", value=(min_d.date(), max_d.date()))
                mask = (filtered["_parsed_date"].dt.date >= start) & (filtered["_parsed_date"].dt.date <= end)
                filtered = filtered.loc[mask]
            filtered = filtered.drop(columns=["_parsed_date"], errors="ignore")

    if category_col and category_col in filtered.columns:
        with st.expander("카테고리 필터"):
            cats = [c for c in sorted(filtered[category_col].dropna().astype(str).unique())]
            selected = st.multiselect("카테고리 선택", options=cats, default=cats)
            filtered = filtered[filtered[category_col].astype(str).isin(selected)]

    return filtered


@st.cache_resource
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# 한국어 감성 분석을 위한 감정 사전
KOREAN_POSITIVE_WORDS = {
    '좋', '좋은', '좋다', '좋아', '좋습니다', '좋아요', '훌륭', '훌륭한', '훌륭하다', '훌륭해', '훌륭합니다',
    '완벽', '완벽한', '완벽하다', '완벽해', '완벽합니다', '완벽해요', '최고', '최고다', '최고입니다', '최고예요',
    '만족', '만족스럽', '만족스러운', '만족스럽다', '만족스러워', '만족스럽습니다', '만족해', '만족합니다',
    '추천', '추천해', '추천합니다', '추천드려', '추천드립니다', '추천해요', '추천해드려요',
    '감사', '감사해', '감사합니다', '감사해요', '고마워', '고마워요', '고맙습니다',
    '사랑', '사랑해', '사랑합니다', '사랑해요', '좋아해', '좋아합니다', '좋아해요',
    '행복', '행복해', '행복합니다', '행복해요', '기쁘', '기쁘다', '기뻐', '기뻐요', '기쁩니다',
    '신나', '신나다', '신나요', '신납니다', '재미있', '재미있다', '재미있어', '재미있어요', '재미있습니다',
    '편리', '편리한', '편리하다', '편리해', '편리합니다', '편리해요', '쉬', '쉽다', '쉬워', '쉬워요', '쉽습니다',
    '빠르', '빠르다', '빨라', '빨라요', '빠릅니다', '빠르게', '빨리', '효율', '효율적', '효율적이다', '효율적이에요',
    '우수', '우수한', '우수하다', '우수해', '우수합니다', '우수해요', '훌륭', '훌륭한', '훌륭하다', '훌륭해',
    '뛰어나', '뛰어난', '뛰어나다', '뛰어나요', '뛰어납니다', '탁월', '탁월한', '탁월하다', '탁월해',
    '훌륭', '훌륭한', '훌륭하다', '훌륭해', '훌륭합니다', '훌륭해요', '훌륭해요', '훌륭해요',
    '정말', '정말로', '진짜', '진심', '완전', '완전히', '너무', '매우', '정말', '정말로', '진짜', '진심',
    'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic', 'awesome', 'perfect', 'best', 'love'
}

KOREAN_NEGATIVE_WORDS = {
    '나쁘', '나쁘다', '나빠', '나빠요', '나쁩니다', '나쁜', '안좋', '안좋다', '안좋아', '안좋아요', '안좋습니다',
    '최악', '최악이다', '최악이에요', '최악입니다', '최악이야', '최악이야요', '최악이에요',
    '불만', '불만스럽', '불만스러운', '불만스럽다', '불만스러워', '불만스럽습니다', '불만이', '불만이에요',
    '화나', '화나다', '화나요', '화납니다', '화가', '화가나', '화가나요', '화가납니다', '짜증', '짜증나', '짜증나요',
    '실망', '실망스럽', '실망스러운', '실망스럽다', '실망스러워', '실망스럽습니다', '실망해', '실망합니다',
    '슬프', '슬프다', '슬퍼', '슬퍼요', '슬픕니다', '우울', '우울하다', '우울해', '우울해요', '우울합니다',
    '힘들', '힘들다', '힘들어', '힘들어요', '힘듭니다', '어렵', '어렵다', '어려워', '어려워요', '어렵습니다',
    '복잡', '복잡한', '복잡하다', '복잡해', '복잡합니다', '복잡해요', '불편', '불편한', '불편하다', '불편해',
    '느리', '느리다', '느려', '느려요', '느립니다', '느리게', '천천히', '비효율', '비효율적', '비효율적이다',
    '문제', '문제가', '문제가있', '문제가있다', '문제가있어', '문제가있어요', '문제가있습니다', '문제있', '문제있다',
    '이상', '이상하다', '이상해', '이상해요', '이상합니다', '이상한', '이상하네', '이상하네요',
    '거짓', '거짓말', '거짓말이', '거짓말이에요', '거짓말입니다', '거짓이', '거짓이에요', '거짓입니다',
    '가짜', '가짜다', '가짜야', '가짜예요', '가짜입니다', '가짜네', '가짜네요',
    '싫', '싫다', '싫어', '싫어요', '싫습니다', '싫어해', '싫어해요', '싫어합니다',
    '혐오', '혐오스럽', '혐오스러운', '혐오스럽다', '혐오스러워', '혐오스럽습니다', '혐오해', '혐오합니다',
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'disappointed', 'angry', 'sad'
}

KOREAN_INTENSIFIERS = {
    '정말', '진짜', '완전', '완전히', '너무', '매우', '정말로', '진심', '정말로', '진짜로', '완전히', '완전',
    '엄청', '엄청나', '엄청나게', '엄청난', '엄청나다', '엄청나요', '엄청납니다',
    '아주', '아주도', '아주도', '아주도', '아주도', '아주도', '아주도', '아주도',
    'so', 'very', 'really', 'extremely', 'absolutely', 'totally', 'completely'
}


def analyze_korean_sentiment(text: str) -> str:
    """한국어 텍스트의 감성을 분석합니다."""
    if not text or pd.isna(text):
        return "중립"
    
    text = str(text).lower().strip()
    if len(text) < 2:
        return "중립"
    
    # 긍정/부정 단어 카운트
    positive_score = 0
    negative_score = 0
    
    # 텍스트를 단어로 분리 (한글, 영문, 숫자 포함)
    words = re.findall(r'[\w가-힣]+', text)
    
    for word in words:
        # 긍정 단어 체크
        if any(pos_word in word for pos_word in KOREAN_POSITIVE_WORDS):
            # 강조어가 앞에 있는지 확인
            word_index = text.find(word)
            if word_index > 0:
                prev_text = text[:word_index].strip()
                if any(intensifier in prev_text for intensifier in KOREAN_INTENSIFIERS):
                    positive_score += 2  # 강조어가 있으면 2점
                else:
                    positive_score += 1
            else:
                positive_score += 1
        
        # 부정 단어 체크
        if any(neg_word in word for neg_word in KOREAN_NEGATIVE_WORDS):
            # 강조어가 앞에 있는지 확인
            word_index = text.find(word)
            if word_index > 0:
                prev_text = text[:word_index].strip()
                if any(intensifier in prev_text for intensifier in KOREAN_INTENSIFIERS):
                    negative_score += 2  # 강조어가 있으면 2점
                else:
                    negative_score += 1
            else:
                negative_score += 1
    
    # 부정 표현 체크 (예: "안 좋다", "별로다", "아니다")
    negative_patterns = [
        r'안\s*좋', r'별로', r'아니', r'못', r'없', r'아닌', r'아닙니다', r'아니에요',
        r'not\s+good', r'not\s+great', r'not\s+excellent', r'not\s+amazing'
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, text):
            negative_score += 1
    
    # 긍정 표현 체크 (예: "정말 좋다", "완전 좋다")
    positive_patterns = [
        r'정말\s*좋', r'완전\s*좋', r'진짜\s*좋', r'너무\s*좋', r'매우\s*좋',
        r'really\s+good', r'very\s+good', r'so\s+good', r'extremely\s+good'
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, text):
            positive_score += 1
    
    # 감성 판정
    if positive_score > negative_score and positive_score > 0:
        return "긍정"
    elif negative_score > positive_score and negative_score > 0:
        return "부정"
    else:
        return "중립"


@st.cache_data(ttl=600)  # 10분 캐시
def analyze_sentiment(texts: List[str], method: str) -> List[str]:
    """텍스트 리스트의 감성을 분석합니다."""
    labels = []
    for text in texts:
        if not text or pd.isna(text):
            labels.append("중립")
            continue
            
        # 한국어 감성 분석 사용
        sentiment = analyze_korean_sentiment(text)
        labels.append(sentiment)
    
    return labels


@st.cache_data(ttl=600)  # 10분 캐시
def extract_keywords(texts: List[str], top_k: int = 20) -> pd.DataFrame:
    # 한국어+영문 토큰 패턴, 한글/영문/숫자 2자 이상
    token_pattern = r"(?u)([\w가-힣]{2,})"
    stop_words = sklearn_text.ENGLISH_STOP_WORDS.union({
        "제품", "사용", "고객", "서비스", "문제", "이슈", "문의", "감사", "정말",
        "합니다", "있어요", "있습니다", "것", "때", "좀", "너무", "매우", "정도"
    })

    vectorizer = TfidfVectorizer(
        token_pattern=token_pattern,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=2
    )
    docs = [str(t) if pd.notna(t) else "" for t in texts]
    if len(docs) == 0:
        return pd.DataFrame(columns=["term", "score"])

    try:
        X = vectorizer.fit_transform(docs)
    except ValueError:
        # 토큰이 없을 때
        return pd.DataFrame(columns=["term", "score"])

    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    df_scores = pd.DataFrame({"term": terms, "score": scores})
    df_scores = df_scores.sort_values("score", ascending=False).head(top_k)
    return df_scores.reset_index(drop=True)


def show_visualizations(df: pd.DataFrame, sentiment_col: str, text_col: str):
    st.subheader("📈 시각화")
    
    # 감성 분포 차트
    col1, col2 = st.columns([2, 1])
    with col1:
        counts = df[sentiment_col].value_counts().reindex(["부정", "중립", "긍정"]).fillna(0).astype(int)
        st.bar_chart(counts)
    
    with col2:
        # 감성 분포 파이 차트를 위한 데이터 준비
        sentiment_data = counts.to_dict()
        st.write("**감성 분포**")
        for sentiment, count in sentiment_data.items():
            percentage = (count / counts.sum()) * 100
            st.write(f"• {sentiment}: {count}개 ({percentage:.1f}%)")

    # 키워드 분석
    with st.expander("🔍 상위 키워드 분석", expanded=True):
        kw_df = extract_keywords(df[text_col].tolist(), top_k=30)
        if len(kw_df) == 0:
            st.info("키워드를 추출할 수 없습니다.")
        else:
            st.dataframe(kw_df, use_container_width=True)
            
            # 키워드 점수 차트
            if len(kw_df) > 0:
                st.bar_chart(kw_df.set_index('term')['score'])


def main():
    st.set_page_config(
        page_title=APP_TITLE, 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 " + APP_TITLE)
    st.caption("💡 CSV 업로드 또는 프로젝트 폴더의 '@feedback-data.csv' 자동 불러오기 지원")
    
    # 앱 설명
    with st.expander("ℹ️ 앱 사용법", expanded=False):
        st.markdown("""
        **이 앱의 기능:**
        1. **데이터 업로드**: CSV 파일을 업로드하거나 기본 파일 사용
        2. **컬럼 매핑**: 텍스트, 날짜, 카테고리 컬럼 선택
        3. **필터링**: 날짜 범위 및 카테고리별 필터링
        4. **감성 분석**: VADER 알고리즘을 사용한 빠른 감성 분석
        5. **키워드 추출**: TF-IDF 기반 상위 키워드 분석
        6. **시각화**: 감성 분포 차트 및 키워드 테이블
        7. **결과 다운로드**: 분석 결과 CSV 파일 다운로드
        """)

    # 데이터 입력 섹션
    with st.sidebar:
        st.header("📁 데이터 입력")
        uploaded = st.file_uploader("CSV 업로드", type=["csv"], help="CSV 파일을 선택하세요")
        default_path = find_default_csv_file()
        if default_path:
            st.success(f"✅ 기본 파일 감지: {DEFAULT_FILE_NAME}")
        else:
            st.info("💡 프로젝트 폴더에 '@feedback-data.csv'를 두면 자동으로 로드됩니다")
        
        st.divider()
        st.header("⚙️ 분석 설정")
        method = st.selectbox("감성 분석 방법", ["한국어 감성 분석 (경량)"], help="한국어에 최적화된 감성 분석 알고리즘입니다")
        run_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)

    df: Optional[pd.DataFrame] = None
    source_label = None
    if uploaded is not None:
        try:
            df = load_dataframe(uploaded)
            source_label = "업로드 파일"
        except Exception as e:
            st.error(f"업로드 파일을 읽는 중 오류: {e}")
    elif default_path:
        try:
            df = load_dataframe(default_path)
            source_label = DEFAULT_FILE_NAME
        except Exception as e:
            st.error(f"기본 파일을 읽는 중 오류: {e}")

    if df is None:
        st.info("📋 좌측에서 CSV를 업로드하거나, 프로젝트 폴더에 '@feedback-data.csv'를 두세요.")
        return

    # 데이터 로딩 성공 메시지
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.success(f"✅ 데이터 로딩 완료 ({source_label})")
    with col2:
        st.metric("총 행 수", f"{len(df):,}")
    with col3:
        st.metric("총 컬럼 수", len(df.columns))
    
    # 데이터 미리보기
    with st.expander("📊 데이터 미리보기", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    text_col, date_col, category_col = ensure_text_column_ui(df)
    df_filtered = apply_filters(df, date_col, category_col)

    if run_btn:
        if text_col not in df_filtered.columns:
            st.error("유효한 텍스트 컬럼을 선택해 주세요.")
            return

        with st.spinner("🔄 감성 분석 중..."):
            sentiments = analyze_sentiment(df_filtered[text_col].tolist(), method)
            result_df = df_filtered.copy()
            result_df = result_df.assign(_sentiment=sentiments)

        st.success("✅ 분석 완료!")
        
        # 분석 결과 요약
        sentiment_counts = result_df['_sentiment'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("긍정", sentiment_counts.get('긍정', 0))
        with col2:
            st.metric("중립", sentiment_counts.get('중립', 0))
        with col3:
            st.metric("부정", sentiment_counts.get('부정', 0))
        with col4:
            st.metric("총 분석 수", len(result_df))

        st.subheader("📋 분석 결과 미리보기")
        display_cols = [text_col, *([date_col] if date_col else []), *([category_col] if category_col else []), "_sentiment"]
        st.dataframe(result_df[display_cols].head(50), use_container_width=True)

        show_visualizations(result_df, sentiment_col="_sentiment", text_col=text_col)

        # 다운로드
        st.subheader("💾 결과 다운로드")
        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv_buf.getvalue(),
            file_name="feedback_analysis_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    else:
        st.info("🚀 좌측의 '분석 실행' 버튼을 눌러 시작하세요.")


if __name__ == "__main__":
    main()


