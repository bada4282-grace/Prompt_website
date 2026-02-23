from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os

# .env 파일 로드 (로컬 테스트용)
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY가 환경 변수에 설정되지 않았습니다.")

app = FastAPI()

# 완벽한 CORS 허용 설정 (Netlify 프론트엔드 도메인 지정 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스 시 Netlify 도메인(예: "https://your-site.netlify.app")으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    text: str

@app.post("/optimize")
async def optimize_prompt(request: PromptRequest):
    try:
        response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """당신은 최상급 메타 프롬프트 엔지니어입니다.
        사용자의 원시 입력을 분석하여 전문가 수준의 실행 지시문으로 변환하십시오.

        [제약 사항]
        1. 분석(COT): 사용자의 핵심 의도, 필요한 도메인 전문가 페르소나, 누락된 배경 정보, 최종 목표를 묵시적으로 추론하십시오.
        2. 출력 통제: 어떠한 인사말, 부가 설명, 마크다운 코드블록 기호(```) 없이 지정된 [출력 템플릿] 구조만 정확히 반환하십시오.
        3. 명확성: 추상적인 어휘를 배제하고, 구체적이고 실행 가능한 행동 지침(Actionable Steps)으로 Task를 구성하십시오.

        [출력 템플릿]
        # Role
        당신은 [분석된 최적의 전문가 페르소나]입니다.

        # Context
        [사용자 입력을 바탕으로 전문적으로 재구성한 배경 상황 및 제약 조건]

        # Task
        - [단계별 수행해야 할 구체적 작업 1]
        - [단계별 수행해야 할 구체적 작업 2]

        #Constraints:
        # 1. 분석 (COT): 사용자의 핵심 의도, 요구되는 도메인 전문가 페르소나, 누락된 배경 정보, 최종 목표를 시스템 내부적으로 사전 추론할 것.
        # 2. 출력 통제: 인사말, 부가 설명, 마크다운 코드 블록 기호(```)의 생성을 절대 금지함. 오직 지정된 [Output Template] 구조만 반환할 것.
        # 3. 명확성: 추상적인 어휘를 배제하고, 'Task' 영역을 구체적이고 실행 가능한 단계별 지침(Actionable steps)으로 구조화할 것."""
                        },
                        {"role": "user", "content": f"원시 입력: {request.text}"}
                    ],
                    temperature=0.1
                )
        return {"optimized_prompt": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))