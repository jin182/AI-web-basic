import os
from dotenv import load_dotenv
from google import genai  # 기존의 generativeai가 아닌 genai를 import 합니다.

# 1. .env 파일 강제 로드 (기존 캐시 무시)
load_dotenv(override=True)

# 2. 환경 변수에서 키 가져오기 (절대 코드에 직접 적지 않음)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요.")
else:
    try:
        # 3. 최신 SDK 방식으로 클라이언트 초기화
        client = genai.Client(api_key=api_key)

        # 4. 답변 생성 요청
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Hello! 최신 라이브러리와 .env 연동 테스트입니다.'
        )
        print("✅ 완벽하게 성공했습니다! 답변:", response.text)

    except Exception as e:
        print("❌ 오류 발생:", e)