import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import re
from datetime import datetime, timedelta

# 1. API 키 숨김 처리 및 OpenAI 객체 초기화
# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 사용할 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 연결된 OpenAI 객체 생성
client = OpenAI(api_key=OPENAI_API_KEY)

# 토큰 사용량 줄이기
def clean_and_truncate_text(text, max_len=250):
    if not text: return ""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned[:max_len]

# 2. 크롤링 함수
def get_oworld_notices(target_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(target_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        notice_list = soup.select("table.faqTbl tbody tr")
        results = []

        for row in notice_list[:10]:
            title_tag = row.select_one("td.title a")
            if title_tag:
                clean_title = title_tag.get_text(strip=True)
                href = title_tag['href']
                match = re.search(r"fn_move_article\('(\d+)'\)", href)
                
                content_text = ""
                image_urls = []

                if match:
                    article_id = match.group(1)
                    detail_url = f"https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx={article_id}&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
                    detail_response = requests.get(detail_url, headers=headers)
                    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                    td_list = detail_soup.find_all("td")
                    if td_list:
                        content_area = max(td_list, key=lambda td: len(td.get_text(strip=True)) + (len(td.find_all("img")) * 500))
                        content_text = content_area.get_text(separator=" ", strip=True)
                        img_tags = content_area.select('img')
                    else:
                        img_tags = detail_soup.select('img')

                    for img in img_tags:
                        img_src = img.get('src')
                        if img_src and img_src.startswith('/'):
                            img_src = f"https://www.oworld.kr{img_src}"
                        if img_src and "icon" not in img_src and "logo" not in img_src and "btn" not in img_src:
                            image_urls.append(img_src)

                    results.append({
                        "title": clean_title,
                        "text": content_text,
                        "images": image_urls
                    })
        return results
    except Exception as e:
        print(f"크롤링 오류 발생: {e}")
        return []

# 3. AI 분석 함수
def get_urgent_notice_json(notices):
    if not notices:
        return json.dumps({"status": "error", "summary": "데이터 없음", "notices": []}, ensure_ascii=False)
    
    # AI에게 보낼 텍스트 정리 (이전과 동일)
    formatted_notices = ""
    for i, item in enumerate(notices, 1):
        clean_text = clean_and_truncate_text(item.get("text", ""))
        img_url = item['images'][0] if item.get('images') else "없음"
        
        formatted_notices += f"[{i}] 제목: {item['title']}\n"
        formatted_notices += f" - 본문: {clean_text}\n"
        formatted_notices += f" - 이미지 URL: {img_url}\n\n"

    # System Prompt: 역할과 대원칙만 부여
    system_prompt = (
        "너는 대전 오월드 방문자 수 예측 시스템의 데이터 분석가야. "
        "절대로 외부 웹 검색(Web Search)을 수행하지 마. "
        "오직 내가 아래에 넘겨준 공지사항 텍스트와 이미지 URL만 참고해서 판단해."
    )
    
    # User Prompt
    user_prompt = f"""최신 공지사항 목록:
        {formatted_notices}

        [중요 지시사항]
        제공된 공지사항 목록 중 '입장객 수 증가/감소'에 영향을 미칠 중요 이벤트(운휴, 행사, 할인 등)만 추출해.
        아래 조건과 JSON 형식에 '정확하게' 맞춰서 출력해. 앞뒤에 쓸데없는 설명은 절대 붙이지 마.

        1. 이벤트 기간(예: 8/12~8/14)이 주어지면, 해당 기간에 속하는 **모든 개별 날짜(YYYY-MM-DD 형식)**를 Key로 만들어서 동일한 이벤트 내용을 반복해서 넣어줘. 연도가 생략되었다면 2026년으로 간주해.
        2. 이벤트 내용은 매우 간략하게 적어 (예: '기차여행 운휴', '여름축제 및 불꽃쇼').
        3. 해당 날짜들을 종합하여 오늘/이번 주 방문객 수에 미칠 영향을 summary(한 줄 요약)로 작성해.

        [출력 JSON 포맷 예시]
        {{
        "status": "urgent",
        "events_by_date": {{
            "2026-07-30": "펀하우스 운휴",
            "2026-08-12": "기차여행 운휴",
            "2026-08-13": "기차여행 운휴",
            "2026-08-14": "기차여행 운휴",
            "2026-08-15": "여름축제 시작"
        }},
        "summary": "펀하우스 및 기차여행 운휴가 있으나, 8월 15일 여름축제로 방문객 증가가 예상됩니다."
        }}
        """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.6-luna", 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        print(f"\n📊 [토큰 사용량] 입력: {response.usage.prompt_tokens} | 출력: {response.usage.completion_tokens} | 총합: {response.usage.total_tokens}")

        ai_result = json.loads(response.choices[0].message.content)
        return json.dumps(ai_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "summary": f"오류 발생: {str(e)}"}, ensure_ascii=False)
    
# 실행부분
if __name__ == "__main__":
    MY_URL = "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkageList.do?mn=KFS_34_01_09_01&bbscd=9&menucd=916" 
    notices = get_oworld_notices(MY_URL)
    json_result = get_urgent_notice_json(notices)
    print("\n--- AI 방문자 영향 요인 분석 결과 (JSON) ---")
    print(json_result)