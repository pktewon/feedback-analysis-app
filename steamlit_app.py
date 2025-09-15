import os
import io
from typing import Optional, List, Tuple

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text

# Optional heavy deps; load lazily when needed
_transformers_pipeline = None
_vader = None


APP_TITLE = "고객 피드백 분석"
DEFAULT_FILE_NAME = "@feedback-data.csv"


def find_default_csv_file() -> Optional[str]:
    workspace_dir = os.getcwd()
    candidate_path = os.path.join(workspace_dir, DEFAULT_FILE_NAME)
    if os.path.exists(candidate_path):
        return candidate_path
    return None


@st.cache_data(show_spinner=False)
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


def load_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def load_transformers_pipeline():
    global _transformers_pipeline
    if _transformers_pipeline is None:
        from transformers import pipeline
        # 다국어 감성 분류 모델
        _transformers_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
    return _transformers_pipeline


def analyze_sentiment(texts: List[str], method: str) -> List[str]:
    if method == "빠름(경량)":
        analyzer = load_vader()
        labels = []
        for t in texts:
            t = str(t) if pd.notna(t) else ""
            scores = analyzer.polarity_scores(t)
            comp = scores.get("compound", 0.0)
            if comp >= 0.05:
                labels.append("긍정")
            elif comp <= -0.05:
                labels.append("부정")
            else:
                labels.append("중립")
        return labels
    else:
        nlp = load_transformers_pipeline()
        outputs = nlp([str(t) if pd.notna(t) else "" for t in texts], truncation=True)
        labels = []
        for out in outputs:
            # 모델 라벨: NEGATIVE/NEUTRAL/POSITIVE
            lbl = out["label"].upper()
            if "NEG" in lbl:
                labels.append("부정")
            elif "NEU" in lbl:
                labels.append("중립")
            else:
                labels.append("긍정")
        return labels


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
    st.subheader("시각화")
    counts = df[sentiment_col].value_counts().reindex(["부정", "중립", "긍정"]).fillna(0).astype(int)
    st.bar_chart(counts)

    with st.expander("상위 키워드"):
        kw_df = extract_keywords(df[text_col].tolist(), top_k=30)
        if len(kw_df) == 0:
            st.info("키워드를 추출할 수 없습니다.")
        else:
            st.dataframe(kw_df, use_container_width=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("CSV 업로드 또는 프로젝트 폴더의 '@feedback-data.csv' 자동 불러오기 지원")

    # 데이터 입력 섹션
    with st.sidebar:
        st.header("데이터 입력")
        uploaded = st.file_uploader("CSV 업로드", type=["csv"])
        default_path = find_default_csv_file()
        if default_path:
            st.success(f"기본 파일 감지: {DEFAULT_FILE_NAME}")
        method = st.selectbox("감성 분석 방법", ["빠름(경량)", "정확(Transformers)"])
        run_btn = st.button("분석 실행")

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
        st.info("좌측에서 CSV를 업로드하거나, 프로젝트 폴더에 '@feedback-data.csv'를 두세요.")
        return

    st.success(f"데이터 로딩 완료 ({source_label}), 행 수: {len(df):,}")
    st.dataframe(df.head(20), use_container_width=True)

    text_col, date_col, category_col = ensure_text_column_ui(df)
    df_filtered = apply_filters(df, date_col, category_col)

    if run_btn:
        if text_col not in df_filtered.columns:
            st.error("유효한 텍스트 컬럼을 선택해 주세요.")
            return

        with st.spinner("감성 분석 중..."):
            sentiments = analyze_sentiment(df_filtered[text_col].tolist(), method)
            result_df = df_filtered.copy()
            result_df = result_df.assign(_sentiment=sentiments)

        st.subheader("분석 결과 미리보기")
        st.dataframe(result_df[[text_col, *([date_col] if date_col else []), *([category_col] if category_col else []), "_sentiment"]].head(50), use_container_width=True)

        show_visualizations(result_df, sentiment_col="_sentiment", text_col=text_col)

        # 다운로드
        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="결과 CSV 다운로드",
            data=csv_buf.getvalue(),
            file_name="feedback_analysis_results.csv",
            mime="text/csv"
        )
    else:
        st.info("좌측의 '분석 실행' 버튼을 눌러 시작하세요.")


if __name__ == "__main__":
    main()


