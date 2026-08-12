import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import re

# 1. API 키 숨김 처리 및 OpenAI 객체 초기화
# .env 파일에서 환경 변수 불러오기
load_dotenv()

# 사용할 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 연결된 OpenAI 객체 생성
client = OpenAI(api_key=OPENAI_API_KEY)


# 2. 핵심 함수 정의
def get_oworld_notices(target_url):
    """오월드 홈페이지에서 최신 공지사항 제목 10개를 크롤링합니다."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(target_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 'faqTbl' 클래스를 가진 테이블의 내용물(tr)만 확인
        notice_list = soup.select("table.faqTbl tbody tr")

        results = []

        for row in notice_list[:10]: # 최신 글 10개 추출
            title_tag = row.select_one("td.title a")
            
            if title_tag:
                clean_title = title_tag.get_text(strip=True)

                # 게시판 본문 내용 파싱
                href = title_tag['href']
                match = re.search(r"fn_move_article\('(\d+)'\)", href)  # article id 가져오기

                content_text = ""
                image_urls = []

                if match:
                    article_id = match.group(1)

                    detail_url = f"https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx={article_id}&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
                    detail_response = requests.get(detail_url, headers=headers)
                    detail_response.raise_for_status()
                    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                    td_list = detail_soup.find_all("td")
                    content_area = None
                    if td_list:
                        content_area = max(td_list, key=lambda td: len(td.get_text(strip=True)) + (len(td.find_all("img")) * 500))
                    
                    if content_area:
                        content_text = content_area.get_text(separator="\n", strip=True)
                        img_tags = content_area.select('img')

                    else:
                        img_tags = detail_soup.select('img')

                    for img in img_tags:
                        img_src = img.get('src')
                        if img_src:
                            if img_src.startswith('/'):
                                img_src = f"https://www.oworld.kr{img_src}"

                            if "icon" not in img_src and "logo" not in img_src and "btn" not in img_src:
                                image_urls.append(img_src)

                    results.append({
                        "title": clean_title,
                        "text": content_text,   # 본문
                        "images": image_urls    # 이미지
                    })
                    
        return results
    except Exception as e:
        print(f"크롤링 오류 발생: {e}")
        return []

def get_urgent_notice_json(target_url, notices):
    """
    크롤링한 공지사항을 OpenAI로 분석하여 JSON 형태로 반환합니다.
    (Streamlit 팀에서 이 함수의 리턴값을 받아서 사용)
    """
    # 1. 크롤링으로 최신 제목들 가져오기
    titles = get_oworld_notices(target_url)
    
    # 크롤링 실패 시 기본 JSON 반환
    if not titles:
        return json.dumps({
            "status": "error",
            "summary": "공지사항 데이터를 불러오지 못했습니다.",
            "notices": []
        }, ensure_ascii=False)
    
    # 2. 리스트를 하나의 문자열로 묶기
    titles_text = "\n".join([f"- {t}" for t in titles])
    
    # 3. 프롬프트에서 JSON 구조와 핵심 키워드 지시
    system_prompt = (
    "너는 대전 오월드 방문자 수 예측 시스템의 데이터 분석가야. "
    "제공된 공지사항 목록 중 '입장객 수(방문자 수) 증가 또는 감소에 직접적인 영향을 미칠 만한 요소'만 선별해. "
    "(예: 특정 놀이기구/시설 운휴, 야간개장, 불꽃쇼, 특별 할인 프로모션, 물놀이장 미운영 등) "

    "조건:"
    "1. 자잘한 행사나 영향이 적은 소식은 무시하고, 방문객 수 변동에 중요한 요인만 종합해."
    "2. 반드시 한 문장(한 줄)으로 요약해."
    "3. 방문객 영향 요소가 전혀 없다면 '현재 방문자 수에 영향을 줄 만한 특이 운휴 및 이벤트가 없습니다.'라고 답해."

    "응답 형식(JSON): {\"status\": \"urgent\" 또는 \"info\" 또는 \"normal\", \"summary\": \"한 줄 요약 내용\"}"
    )
    
    try:
        # 미리 만들어둔 client 객체를 사용해 통신
        response = client.chat.completions.create(
            model="gpt-5.5",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"최신 공지사항 목록:\n{notices}"}
            ],
        )
        
        # AI가 만들어준 JSON 문자열을 딕셔너리로 변환
        ai_result = json.loads(response.choices[0].message.content)
        
        # 최종 JSON 문자열로 예쁘게 변환하여 반환
        return json.dumps(ai_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "summary": f"AI 분석 중 오류 발생: {str(e)}",
            "notices": titles
        }, ensure_ascii=False)

# 테스트용 메인 실행부
if __name__ == "__main__":
    MY_URL = "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkageList.do?mn=KFS_34_01_09_01&bbscd=9&menucd=916" 

    notices = get_oworld_notices(MY_URL)
    ''' 디버깅
    for i, notice in enumerate(notices, 1):
        print(f"[{i}] {notice['title']}")
        print(f" - 본문: {notice['text']}")
        print(f" - 이미지 수: {len(notice['images'])}장")
        print(f" - 이미지 URL: {notice['images']}")
        print("-" * 40)
    '''
    json_result = get_urgent_notice_json(MY_URL, notices)
    print("\n--- AI 방문자 영향 요인 분석 결과 (JSON) ---")
    print(json_result)