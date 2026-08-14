import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime

# 1. API 키 숨김 처리 및 OpenAI 객체 초기화
# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 사용할 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 연결된 OpenAI 객체 생성
client = OpenAI(api_key=OPENAI_API_KEY)

def clean_and_truncate_text(text, max_len=250):
    """
    [데이터 전처리 및 토큰 최적화 함수]
    크롤링된 HTML 원문 텍스트의 불필요한 공백과 줄바꿈을 압축하고, 
    정해진 길이(max_len)만큼만 잘라내어 OpenAI 입력 토큰 비용을 최소화
    """
    if not text: return ""
    # \s+ 정규식을 이용해 탭, 연속된 띄어쓰기, 줄바꿈을 단일 공백으로 치환
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned[:max_len]

# 2. 크롤링 함수
def get_oworld_notices(target_url):
    """
    [오월드 공지사항 웹 크롤러]
    최신 공지사항 목록에서 제목과 상세 URL을 추출하고, 
    해당 URL로 다시 접근하여 본문 텍스트를 파싱하여 반환
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    results = []
    
    try:
        response = requests.get(target_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 'faqTbl' 클래스를 가진 테이블의 내용물(tr)만 확인
        notice_list = soup.select("table.faqTbl tbody tr")

        # 비용 절감 및 데이터 효율을 위해 최신 공지 7개만 수
        for row in notice_list[:7]:
            title_tag = row.select_one("td.title a")
            if title_tag:
                clean_title = title_tag.get_text(strip=True)
                # 게시판 본문 내용 파싱
                href = title_tag['href']
                # Javascript 함수 내부에 숨겨진 게시글 ID(bbsIdx) 정규식 추출
                match = re.search(r"fn_move_article\('(\d+)'\)", href)
                
                content_text = ""
                detail_url = target_url
                
                if match:
                    article_id = match.group(1)
                    # 실제 게시글 상세 페이지 URL 조합
                    detail_url = f"https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx={article_id}&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
                    
                    detail_response = requests.get(detail_url, headers=headers)
                    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                    
                    td_list = detail_soup.find_all("td")
                    if td_list:
                        # 텍스트가 가장 많은 <td> 영역을 본문으로 간주하여 추출
                        content_area = max(td_list, key=lambda td: len(td.get_text(strip=True)))
                        content_text = content_area.get_text(separator=" ", strip=True)

                results.append({
                    "title": clean_title,
                    "text": content_text,
                    "url": detail_url
                })
        return results
    except Exception as e:
        print(f"크롤링 오류 발생: {e}")
        return []

# 3. AI 분석 함수 (URL 직접 주입 방식 적용)
def get_urgent_notice_json(notices):
    """
    [AI 기반 공지사항 구조화 에이전트]
    수집된 텍스트를 바탕으로 UI 시안에 맞춘 표준 JSON 데이터를 생성
    URL 임의 생성(환각) 방지를 위해 원본 데이터를 프롬프트에 직접 주입
    """
    if not notices:
        # 시스템 에러 방지를 위한 빈 데이터 Fallback(기본값) 반환
        return json.dumps({
            "status": "error", 
            "notices_list": [], 
            "event_board_url": ""
        }, ensure_ascii=False)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # [환각(Hallucination) 차단] 
    # AI가 링크를 지어내지 못하도록 [제목, 본문, 원본 URL]을 하나의 세트로 묶어서 주입
    formatted_text = "[최신 공지사항 7개 데이터]\n"
    for i, item in enumerate(notices, 1):
        clean_text = clean_and_truncate_text(item.get("text", ""))
        formatted_text += f"[공지 {i}]\n- 원본 제목: {item['title']}\n- 본문 요약: {clean_text}\n- URL: {item['url']}\n\n"

    # 프롬프트를 '전체 규칙'과 '세부 항목 규칙'으로 분리하여 JSON 스키마 붕괴 방지
    user_prompt = f"""{formatted_text}
        [중요 지시사항]
        오늘 날짜는 {today_str}입니다. 
        제공된 공지사항 데이터를 분석하여 UI 시안과 완벽히 일치하는 형태의 JSON을 만들어.
        가장 중요도가 높은 공지사항 최대 5개만 선별해.

        [전체 JSON 구조 규칙]
        1. major_summary: 추출된 데이터들을 종합하여, 오늘부터 10일간 오월드 방문객 수(혼잡도)에 미칠 영향을 1~2줄로 브리핑해. (최상단에 딱 1번만 출력)
        2. event_board_url: 무조건 "https://www.oworld.kr/newkfsweb/kfi/kfs/event/selectDccoEventList.do?mn=KFS_34_02_03_01" 이 주소를 고정으로 출력해. 절대 비우지 마.

        [notices_list 내부 각 항목(item) 규칙]
        1. category: ['축제', '행사', '점검', '안내', '예매'] 중 하나로 분류.
        2. title: 원본 제목을 깔끔하게 다듬어서 작성.
        3. date_string: 본문에서 날짜를 찾아 'MM.DD' 또는 'MM.DD ~ MM.DD' 형식으로 작성 (없으면 '상시').
        4. status_tag: 오늘({today_str}) 기준 행사 상태 (진행 중, D-Day, D-X).
        5. url: 내가 위에 제공한 텍스트에서 해당 공지의 'URL'을 그대로 복사해서 붙여넣어.

        [출력 JSON 포맷 예시]
        {{
        "status": "success",
        "major_summary": "8월 15일부터 여름축제와 야간개장이 시작되어 다가오는 주말 방문객이 집중될 것으로 예상되며, 펀하우스 등 일부 시설은 당분간 운휴합니다.",
        "notices_list": [
            {{
            "category": "점검",
            "title": "회전목마·바이킹 정기 안전점검 운휴",
            "date_string": "08.19 ~ 08.20",
            "status_tag": "D-6",
            "url": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx=..."
            }}
        ],
        "event_board_url": "https://www.oworld.kr/newkfsweb/kfi/kfs/event/selectDccoEventList.do?mn=KFS_34_02_03_01"
        }}
        """

    system_prompt = "너는 대전 오월드 웹사이트의 데이터 가공을 담당하는 AI야. 철저히 주어진 텍스트 기반으로 답변하고, 특히 URL은 절대로 창작하지 말고 원본을 그대로 써."
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.6-luna", 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt} 
            ]
        )

        # 유지보수 및 비용 모니터링을 위한 토큰 사용량 확인
        print(f"[토큰 사용량] 입력: {response.usage.prompt_tokens} | 출력: {response.usage.completion_tokens} | 총합: {response.usage.total_tokens}")

        # AI가 만든 JSON을 그대로 반환
        return response.choices[0].message.content
        
    except Exception as e:
        return json.dumps({"status": "error", "notices_list": [], "event_board_url": ""}, ensure_ascii=False)

# 4. 실행부분
if __name__ == "__main__":
    MY_URL = "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkageList.do?mn=KFS_34_01_09_01&bbscd=9&menucd=916" 
    
    # 1) 크롤링 실행
    notices_data = get_oworld_notices(MY_URL) 
    
    # 2) AI 요약 (UI 시안 맞춤형)
    json_result = get_urgent_notice_json(notices_data) 
    
    print("\n--- UI 팀 전달용 최종 JSON ---")
    print(json_result)