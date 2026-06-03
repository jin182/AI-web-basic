import os
from google import genai

# 클라이언트 초기화
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 최신 모델로 변경 (gemini-2.5-flash 또는 gemini-3.5-flash 사용 가능)
response = client.models.generate_content(
    model="gemini-2.5-flash",  # <-- 이 부분을 2.5 또는 3.5로 수정하세요!
    contents="안녕하세요"
)

print(response.text)