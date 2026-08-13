 # 함수 호출 사용 예시
  from oworld_openai_pipeline import (
    build_ui_payload,
    get_selected_day
)

payload = build_ui_payload(days=10)

selected_day = get_selected_day(
    payload,
    "2026-08-15"
)
  
  # 테스트용 3일 출력 결과
[토큰 사용량] 입력: 1815 | 출력: 1470 | 총합: 3285
{
  "generated_at": "2026-08-13T18:03:42+09:00",
  "weather_source": "Open-Meteo / 대전 오월드 좌표 / 낮 12시 예보",
  "notice_source": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkageList.do?mn=KFS_34_01_09_01&bbscd=9&menucd=916",
  "feature_columns": [
    "dayoff",
    "nextdayoff",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "temperature",
    "rain",
    "humidity"
  ],
  "notice": {
    "status": "success",
    "major_summary": "8월 15일부터 여름축제와 야간개장·불꽃쇼가 시작되어 8/15~8/22 방문객이 증가할 것으로 예상됩니다. 8월 14일까지 학생 할인 프로모션이 진행되며, 일부 놀이시설은 운휴 중입니다.",
    "notices_list": [
      {
        "category": "축제",
        "title": "오월드 여름축제 오픈",
        "date_string": "08.15",
        "status_tag": "D-2",
        "url": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx=25202&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
      },
      {
        "category": "행사",
        "title": "오월드 8월 야간개장 및 불꽃쇼 안내",
        "date_string": "08.15",
        "status_tag": "D-2",
        "url": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx=25199&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
      },
      {
        "category": "점검",
        "title": "놀이시설 기차여행 운휴 안내",
        "date_string": "08.12 ~ 08.14",
        "status_tag": "진행 중",
        "url": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx=25203&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
      },
      {
        "category": "점검",
        "title": "놀이시설 펀하우스 운휴 안내",
        "date_string": "07.30 ~ 08.28",
        "status_tag": "진행 중",
        "url": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx=25198&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
      },
      {
        "category": "예매",
        "title": "오월드 여름방학 특별 할인 프로모션 ‘썸머틴’",
        "date_string": "07.20 ~ 08.14",
        "status_tag": "진행 중",
        "url": "https://www.oworld.kr/newkfsweb/kfi/kfs/linkage/selectDccoLinkage.do?bbsIdx=25191&mn=KFS_34_01_09_01&bbscd=9&menucd=916"
      }
    ],
    "event_board_url": "https://www.oworld.kr/newkfsweb/kfi/kfs/event/selectDccoEventList.do?mn=KFS_34_02_03_01"
  },
  "days": [
    {
      "date": "2026-08-13",
      "status": "success",
      "data": {
        "holiday": false,
        "weekend": false,
        "weekday": {
          "mon": false,
          "tue": false,
          "wed": false,
          "thur": true,
          "fri": false,
          "sat": false,
          "sun": false
        },
        "temperature": 28.5,
        "rain": 47.0,
        "humidity": 65.0,
        "predicted_visitors": 764,
        "congestion": 0,
        "user_score": 36,
        "weather_score": 76
      },
      "meta": {
        "weather_summary": "맑음",
        "weather_icon": "☀️",
        "weather_source": "Open-Meteo",
        "forecast_time": "12:00",
        "model_dayoff": false,
        "model_nextdayoff": false,
        "model_month": "08"
      }
    },
    {
      "date": "2026-08-14",
      "status": "success",
      "data": {
        "holiday": false,
        "weekend": false,
        "weekday": {
          "mon": false,
          "tue": false,
          "wed": false,
          "thur": false,
          "fri": true,
          "sat": false,
          "sun": false
        },
        "temperature": 28.1,
        "rain": 49.0,
        "humidity": 75.0,
        "predicted_visitors": 657,
        "congestion": 0,
        "user_score": 38,
        "weather_score": 71
      },
      "meta": {
        "weather_summary": "구름 조금",
        "weather_icon": "🌤️",
        "weather_source": "Open-Meteo",
        "forecast_time": "12:00",
        "model_dayoff": false,
        "model_nextdayoff": true,
        "model_month": "08"
      }
    },
    {
      "date": "2026-08-15",
      "status": "success",
      "data": {
        "holiday": true,
        "weekend": true,
        "weekday": {
          "mon": false,
          "tue": false,
          "wed": false,
          "thur": false,
          "fri": false,
          "sat": true,
          "sun": false
        },
        "temperature": 27.0,
        "rain": 12.0,
        "humidity": 76.0,
        "predicted_visitors": 4228,
        "congestion": 2,
        "user_score": 18,
        "weather_score": 83
      },
      "meta": {
        "weather_summary": "구름 조금",
        "weather_icon": "🌤️",
        "weather_source": "Open-Meteo",
        "forecast_time": "12:00",
        "model_dayoff": true,
        "model_nextdayoff": true,
        "model_month": "08"
      }
    }
  ]
}
