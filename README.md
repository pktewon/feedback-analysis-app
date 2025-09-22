# 📊 고객 피드백 분석 (Streamlit)

경량화된 고객 피드백 감성 분석 웹 애플리케이션입니다. 한국어에 최적화된 감성 분석 알고리즘을 사용하여 빠르고 정확하게 텍스트 데이터를 분석합니다.

## ✨ 주요 기능

- 🔍 **감성 분석**: 한국어에 최적화된 감성 분석 알고리즘
- 📈 **키워드 추출**: TF-IDF 기반 상위 키워드 분석
- 🎯 **데이터 필터링**: 날짜 범위 및 카테고리별 필터링
- 📊 **시각화**: 감성 분포 차트 및 키워드 분석
- 💾 **결과 다운로드**: 분석 결과 CSV 파일 저장
- ⚡ **경량화**: Streamlit Cloud 배포 최적화

## 🚀 로컬 실행 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd job_skill_project
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 앱 실행
```bash
streamlit run app.py
```

### 5. 데이터 준비
- 프로젝트 폴더에 `@feedback-data.csv` 파일을 두거나
- 웹 인터페이스에서 CSV 파일을 업로드

## ☁️ Streamlit Cloud 배포

### 1. GitHub에 코드 업로드
1. GitHub 저장소 생성
2. 코드 업로드 (app.py, requirements.txt, README.md)

### 2. Streamlit Cloud에서 배포
1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. "New app" 클릭
3. GitHub 저장소 연결
4. 메인 파일 경로: `app.py`
5. "Deploy" 클릭

### 3. 배포 후 설정
- 앱이 자동으로 빌드되고 배포됩니다
- URL이 생성되어 공유 가능합니다
- 데이터는 웹 인터페이스에서 업로드하여 사용

## 📋 데이터 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:
- **텍스트 컬럼**: 분석할 텍스트 데이터 (필수)
- **날짜 컬럼**: 날짜 정보 (선택사항)
- **카테고리 컬럼**: 제품군/카테고리 정보 (선택사항)

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python
- **감성 분석**: 한국어 감정 사전 기반 분석
- **텍스트 처리**: scikit-learn (TF-IDF)
- **데이터 처리**: pandas

## 📦 의존성

```
streamlit==1.37.1
pandas==2.2.2
scikit-learn==1.5.1
vaderSentiment==3.3.2
```

## 🔧 최적화 사항

- **경량화**: transformers, torch 등 무거운 의존성 제거
- **한국어 최적화**: 한국어 감정 사전 기반 정확한 감성 분석
- **캐싱**: @st.cache_data, @st.cache_resource로 성능 최적화
- **사용자 경험**: 직관적인 UI/UX 개선
- **배포 최적화**: Streamlit Cloud 호환성 확보

## 📞 지원

문제가 발생하거나 개선 사항이 있으시면 이슈를 등록해 주세요.

