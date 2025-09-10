# 고객 피드백 분석 (Streamlit)

## 실행 방법

1) 가상환경(선택) 생성 후 활성화
```bash
python -m venv .venv
. .venv/Scripts/activate
```

2) 의존성 설치
```bash
pip install -r requirements.txt
```

3) 앱 실행
```bash
streamlit run app.py
```

4) 데이터
- 프로젝트 폴더에 `@feedback-data.csv`가 있으면 자동 로딩됩니다.
- 사이드바에서 CSV를 업로드해도 동작합니다.

## 기능
- 텍스트 감성 분석: 경량(VADER) 또는 고정밀(Transformers)
- 키워드 추출: TF-IDF 기반 상위 키워드 테이블 제공
- 필터: 날짜 범위, 카테고리(제품군) 선택
- 결과 다운로드: 분석 결과 CSV 저장

## 참고
- Transformers 모델은 최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다.

