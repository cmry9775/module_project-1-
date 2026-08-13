"""theme.py - 폰트/색상 등 화면 스타일 상수 모음."""

# 혼잡도 라벨 → 색상. ML팀이 새 라벨을 주더라도 죽지 않도록
# app.py 에서는 반드시 LEVEL_COLOR.get(level, LEVEL_COLOR_FALLBACK) 로 접근한다.
LEVEL_COLOR = {
    "여유": "#A8CBB0",
    "보통": "#E8C79A",
    "혼잡": "#E7A9A0",
}
LEVEL_COLOR_FALLBACK = "#BFC7D1"

# 추천 1순위 막대 강조색
BEST_COLOR = "#7BA7D4"

# 공지 카테고리 → 색상. data_provider.NOTICES 의 category 와 짝을 이룬다.
# 새 카테고리가 들어와도 죽지 않도록 NOTICE_COLOR.get(..., NOTICE_COLOR_FALLBACK) 로 쓴다.
NOTICE_COLOR = {
    "축제": "#E7A9A0",
    "행사": "#7BA7D4",
    "점검": "#E8C79A",
    "예매": "#A8CBB0",
    "안내": "#B0AEC9",
}
NOTICE_COLOR_FALLBACK = "#BFC7D1"

# 공지 진행 상태 → 글자색. '진행 중'은 초록, '예정(D-n)'은 파랑으로 구분한다.
NOTICE_STATE_COLOR = {
    "ongoing": "#2F7A57",
    "upcoming": "#2F6FD0",
    "always": "#6B7280",
    "ended": "#9AA1AC",
}
NOTICE_STATE_COLOR_FALLBACK = "#6B7280"

# 그래프 x축 날짜 레이블 색. 달력처럼 토요일은 파랑, 일요일·공휴일은 빨강.
DATE_LABEL_COLOR = "#1F2A37"
SATURDAY_LABEL_COLOR = "#2F6FD0"
HOLIDAY_LABEL_COLOR = "#D8453F"

# 기상 수치 범위별 색 (사이드바 등). 쾌적 → 주의로 갈수록 따뜻/강한 톤.
def temp_color(temp: float) -> str:
    if temp <= 5:
        return "#4A7FB5"   # 한파
    if temp <= 14:
        return "#7BA7D4"   # 쌀쌀
    if temp <= 27:
        return "#6BAF8D"   # 쾌적
    if temp <= 32:
        return "#E0A060"   # 더움
    return "#D98B80"       # 폭염


def rain_color(rain_prob: float) -> str:
    if rain_prob <= 20:
        return "#6BAF8D"   # 맑음에 가까움
    if rain_prob <= 40:
        return "#7BA7D4"   # 낮은 비 소식
    if rain_prob <= 60:
        return "#E0A060"   # 비 가능성
    return "#5B8DB8"       # 비 확률 높음


def score_color(score: float) -> str:
    """추천 점수 0~100 구간색. 낮음(적) → 보통(황) → 높음(녹)."""
    if score >= 70:
        return "#6BAF8D"
    if score >= 40:
        return "#E0A060"
    return "#D98B80"


def humidity_color(humidity: float) -> str:
    if humidity < 40:
        return "#E0A060"   # 건조
    if humidity < 70:
        return "#6BAF8D"   # 쾌적
    if humidity < 80:
        return "#E0A060"   # 다소 습함
    return "#D98B80"       # 매우 습함

FONT_OPTIONS = {
    "Pretendard": "Pretendard",
    "에스코어 드림": "Escoredream",
    "G마켓 산스": "GMarketSans",
    "원스토어 모바일": "OneStoreMobileGothicBody",
}

FONT_FACE_CSS = """
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Light.woff2') format('woff2');
    font-weight: 300; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Regular.woff2') format('woff2');
    font-weight: 400; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Medium.woff2') format('woff2');
    font-weight: 500; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-SemiBold.woff2') format('woff2');
    font-weight: 600; font-display: swap;
}
@font-face {
    font-family: 'Pretendard';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/pretendard@1.0/Pretendard-Bold.woff2') format('woff2');
    font-weight: 700; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-3Light.woff') format('woff');
    font-weight: 300; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-4Regular.woff') format('woff');
    font-weight: 400; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-5Medium.woff') format('woff');
    font-weight: 500; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-6Bold.woff') format('woff');
    font-weight: 600; font-display: swap;
}
@font-face {
    font-family: 'Escoredream';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_six@1.2/S-CoreDream-7ExtraBold.woff') format('woff');
    font-weight: 700; font-display: swap;
}
@font-face {
    font-family: 'GMarketSans';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansLight.woff') format('woff');
    font-weight: 300; font-display: swap;
}
@font-face {
    font-family: 'GMarketSans';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansMedium.woff') format('woff');
    font-weight: 500; font-display: swap;
}
@font-face {
    font-family: 'GMarketSans';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansBold.woff') format('woff');
    font-weight: 700; font-display: swap;
}
@font-face {
    font-family: 'OneStoreMobileGothicBody';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2105_2@1.0/ONE-Mobile-Regular.woff') format('woff');
    font-weight: normal; font-display: swap;
}
"""
