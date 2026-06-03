import os
import json
import time
import logging
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# RAG 관련 라이브러리 (선택적 로드)
try:
    import numpy as np
    from scipy.sparse import load_npz
    from sklearn.metrics.pairwise import cosine_similarity
    from joblib import load
    from google import genai  
    from google.genai import types  
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG 라이브러리 누락: {e}")
    RAG_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__)
CORS(app, origins=["https://jin182.github.io/AI-web-basic-front/"], supports_credentials=True )
app.config['SECRET_KEY'] = 'legal-rag-secret-2024'

# 전역 변수
DATABASE_PATH = "legal_rag.db"
rag_system = {
    "available": RAG_AVAILABLE,
    "loaded": False,
    "client": None,      
    "X": None,
    "vectorizer": None,
    "chunks": None,
    "meta": None,
    "error": None
}

# ======================== 데이터베이스 관리 ========================

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """데이터베이스 초기화 및 마이그레이션"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'")
            existing_sessions = cursor.fetchone()
            
            if existing_sessions and 'token' not in existing_sessions[0]:
                logger.warning("기존 sessions 테이블이 구버전입니다. 재생성합니다.")
                cursor.execute("DROP TABLE IF EXISTS sessions")
                cursor.execute("DROP TABLE IF EXISTS messages")
                cursor.execute("DROP TABLE IF EXISTS chat_sessions")
                
        except Exception as e:
            logger.info(f"테이블 확인 중: {e}")
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 세션 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 채팅 세션 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 메시지 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_session_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                answer_type TEXT DEFAULT 'combined',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_session_id) REFERENCES chat_sessions (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN answer_type TEXT DEFAULT 'combined'")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        logger.info("✅ 데이터베이스 초기화 완료")

# ======================== 인증 함수 ========================

def hash_password(password):
    salt = secrets.token_hex(16)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{hash_obj.hex()}"

def verify_password(password, hashed):
    try:
        salt, stored_hash = hashed.split(':')
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hash_obj.hex() == stored_hash
    except:
        return False

def create_session_token():
    return secrets.token_urlsafe(32)

def get_user_from_token(token):
    if not token:
        return None
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN sessions s ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > datetime('now')
        ''', (token,))
        return cursor.fetchone()

# ======================== 개선된 AI + RAG 시스템 ========================

def load_rag_system():
    """RAG 시스템 로드"""
    if not RAG_AVAILABLE:
        rag_system["error"] = "RAG 라이브러리가 설치되지 않았습니다"
        return False
    
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            rag_system["error"] = "GEMINI_API_KEY 환경변수가 설정되지 않았습니다"
            return False
        
        rag_system["client"] = genai.Client(api_key=api_key)
        logger.info("✅ Gemini 신규 SDK Client 설정 완료")
        
        possible_paths = [
            Path("./index"),
            Path("../index"), 
            Path("index"),
            Path(os.environ.get("INDEX_DIR", "./index"))
        ]
        
        index_dir = None
        for path in possible_paths:
            if path.exists() and (path / "tfidf_matrix.npz").exists():
                index_dir = path
                break
        
        if not index_dir:
            rag_system["error"] = "RAG 인덱스 파일을 찾을 수 없습니다"
            logger.warning("⚠️ 인덱스 파일을 찾을 수 없습니다")
            return False
        
        logger.info(f"📁 인덱스 로드 중: {index_dir.absolute()}")
        
        X = load_npz(index_dir / "tfidf_matrix.npz")
        vectorizer = load(index_dir / "vectorizer.joblib")
        
        chunks = []
        with open(index_dir / "chunks.txt", "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(line.rstrip("\n").replace("\\n", "\n"))
        
        meta = []
        with open(index_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    meta.append(json.loads(line))
        
        assert len(chunks) == X.shape[0] == len(meta), "인덱스 크기 불일치"
        
        rag_system["X"] = X
        rag_system["vectorizer"] = vectorizer
        rag_system["chunks"] = chunks
        rag_system["meta"] = meta
        rag_system["loaded"] = True
        
        sources = list(set(m["source"] for m in meta))
        logger.info(f"✅ RAG 시스템 로드 완료 (청크: {len(chunks)}개)")
        
        return True
        
    except Exception as e:
        rag_system["error"] = f"RAG 시스템 로드 실패: {str(e)}"
        logger.error(f"❌ RAG 로드 오류: {e}")
        return False

def search_chunks(query, k=5):
    """RAG 검색 함수"""
    if not rag_system["loaded"]:
        return []
    
    try:
        X = rag_system["X"]
        vectorizer = rag_system["vectorizer"]
        chunks = rag_system["chunks"]
        meta = rag_system["meta"]
        
        q_vec = vectorizer.transform([query])
        if q_vec.nnz == 0:
            return []
        
        similarities = cosine_similarity(X, q_vec).ravel()
        top_indices = np.argsort(-similarities)[:k]
        
        results = []
        for idx in top_indices:
            sim_score = similarities[idx]
            if sim_score > 0.01:
                results.append({
                    "index": int(idx),
                    "similarity": float(sim_score),
                    "source": meta[idx]["source"],
                    "chunk_id": meta[idx]["chunk_id"],
                    "article_numbers": meta[idx].get("article_numbers", []),
                    "isms_controls": meta[idx].get("isms_controls", []),
                    "text": chunks[idx]
                })
        return results
    except Exception as e:
        logger.error(f"RAG 검색 오류: {e}")
        return []

def check_legal_relevance(question):
    """질문이 도메인(보안, 법률, CPPG, ISMS-P) 관련인지 검증"""
    legal_keywords = [
        '법', '법률', '개인정보', '보호법', '정보통신', '망법', 'isms', 'isms-p', 'cppg',
        '인증', '심사', '통제', '항목', '수집', '동의', '제3자', '위탁', '파기', '암호화',
        '안전성', '확보', '조치', '고시', '기준', '과태료', '과징금', '유출', '신고',
        '정보보호', '최고책임자', 'ciso', 'cpo', '가명', '익명', '결함', '사례'
    ]
    
    question_lower = question.lower()
    for keyword in legal_keywords:
        if keyword in question_lower:
            return True
    return False

def generate_combined_answer(question, rag_context=""):
    """수정된 프롬프트 엔지니어링: CPPG & ISMS-P 수험 및 실무 특화 프롬프트 라우팅"""
    if not rag_system["client"]:
        raise ValueError("Gemini API Client가 초기화되지 않았습니다")
    
    is_legal = check_legal_relevance(question)
    current_date = datetime.now().strftime('%Y년 %m월 %d일')
    
    # 1. RAG 내부 백서/인덱스 컨텍스트가 확보된 경우
    if rag_context.strip():
        prompt = f"""당신은 대한민국의 최고 권위 있는 '개인정보보호 및 정보보호 관리체계(ISMS-P)' 인증 심사원이자 'CPPG(개인정보관리사)' 수험 전문가입니다. 
답변하기 전에 반드시 관련 최신 고시, 가이드라인 개정 사항, 개인정보보호위원회 자료를 검색하여 사실관계를 검증하세요.

**현재 날짜:** {current_date}
**사용자 질문 (수험생 또는 보안 실무자):** {question}

**RAG 검색된 핵심 준거성 데이터 (우선 신뢰 자료):** 
{rag_context}

**[반드시 준수해야 할 답변 구조 및 원칙]**
1. **실시간 정보 동기화 (웹 검색):** 관련 법령은 개정이 잦으므로, 검색을 통해 질문과 관련된 최신 시행령이나 과징금/과태료 기준 변경이 있는지 먼저 확인하고 반영하세요. 
2. **CPPG 수험적 관점 분석:** 해당 질문이 CPPG 자격증 시험에 출제된다면 핵심 판정 기준(예: 동의를 받아야 하는 필수 요건 vs 동의 제외 예외 원칙)이 무엇인지 명확한 준거 기준을 제시하세요. 
3. **ISMS-P 인증 실무 연계:** 관련된 ISMS-P 인증기준 통제 항목(예: 1.1.1 최고경영자 지정, 3.1.1 개인정보 수집 제한 등)을 명시하고, 실제 심사 시 결함 항목으로 지적될 수 있는 단골 위반 사례를 함께 설명하세요.
4. **가독성 최적화:** 어려운 법령 문장을 통째로 나열하지 말고, 수험생과 실무자가 직관적으로 이해할 수 있도록 구조화된 표(Table)나 가독성 높은 글머리 기호(Bullet points)를 적극 활용하세요. 
5. 살제 CPPG 시험 문제 스타일로 답변을 재구성하여, 핵심 판정 기준과 실무 적용 포인트가 명확히 드러나도록 안내하세요.
6. **출처 명시:** 답변에 인용된 조항(예: 개인정보 보호법 제15조 제1항) 및 ISMS-P 통제 항목 번호를 명확히 구분하여 표기하세요. 
**답변:**"""

    # 2. RAG에 관련 청크는 없지만 보안/개인정보보호 범위에 속하는 질문일 때
    else:
        if is_legal:
            prompt = f"""당신은 정보보호 및 개인정보보호 컨설팅 전문가입니다. 
CPPG 및 ISMS-P 관련 도메인 지식을 바탕으로 성실하게 답변하되, 내부 가이드북(RAG)에 데이터가 없는 상태이므로 웹 검색을 활용해 최신 공공기관(KISA, 개인정보위) 오피셜 가이드를 참고하여 답변하세요.

**현재 날짜:** {current_date}
**질문:** {question}

**[답변 작성 가이드]**
1. 질문과 매핑되는 '개인정보보호법', '정보통신망법', 또는 'ISMS-P 인증기준'의 맥락을 추론하여 한국 법령 기준의 표준 답변을 생성하세요. 
2. 수험생을 위해 시험에 자주 출제되고 헷갈리기 쉬운 유사 개념(예: 개인정보 수집 동의 vs 제3자 제공 동의, 가명정보 vs 익명정보)이 있다면 비교 분석 내용을 포함해 주세요. 
3. 실무적인 조치 사항(구체적인 암호화 조치 방식, 기술적·관리적 보호조치 절차 등)이 있다면 실제 적용 가능한 실무 시나리오 관점에서 안내하세요.
4. 확실하지 않거나 판례가 대립하는 사안은 KISA 상담 센터나 법률 전문가의 유권해석이 필요함을 안내하세요.

**답변:**"""
        
        # 3. 도메인(CPPG, ISMS-P, 보안) 외 완전히 무관한 질문 필터링 라우터
        else:
            prompt = f"""사용자가 다음과 같이 질문했습니다: "{question}"

이 질문은 개인정보보호법, 정보보호, CPPG, ISMS-P 등 본 서비스의 학술/실무 가이드 목적과 전혀 관련이 없는 질문입니다. 

다음 지침에 따라 정중하게 거절 및 안내 메시지를 작성하세요:
1. 본 서비스는 'CPPG 및 ISMS-P 자격증/실무 학습 가이드 챗봇'임을 명확히 안내합니다. 
2. 개인정보보호법 조항, 정보보호 기술적/관리적 보호조치, ISMS-P 인증기준 등 프로젝트 도메인에 맞는 질문을 입력하도록 친절히 유도하세요.
3. 도메인 외 무관한 일상 대화나 타 분야 질문에는 답변할 수 없음을 정중히 양해 구하세요."""

    try:
        # 최신 안정화 모델인 gemini-2.5-flash 및 고도화된 웹 검색 도구 세팅
        response = rag_system["client"].models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],  
                temperature=0.2,
                max_output_tokens=20000,
                top_p=0.9
            )
        )
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini 답변 생성 오류: {e}")
        try:
            response = rag_system["client"].models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=20000
                )
            )
            return response.text.strip()
        except Exception as e2:
            logger.error(f"Fallback 시도 실패: {e2}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# ======================== API 엔드포인트 ========================

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()  
        password = data.get('password', '').strip()
        
        if not all([username, email, password]):
            return jsonify({"error": "모든 필드를 입력해주세요"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "비밀번호는 6자 이상이어야 합니다"}), 400
        
        hashed = hash_password(password)
        
        with get_db() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, hashed)
                )
                conn.commit()
                logger.info(f"새 사용자 등록: {username}")
                return jsonify({"message": "회원가입 성공"}), 201
            except sqlite3.IntegrityError:
                return jsonify({"error": "이미 존재하는 사용자명 또는 이메일"}), 409
                
    except Exception as e:
        logger.error(f"회원가입 오류: {e}")
        return jsonify({"error": "회원가입 처리 실패"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not all([username, password]):
            return jsonify({"error": "사용자명과 비밀번호를 입력해주세요"}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if not user or not verify_password(password, user['password_hash']):
                return jsonify({"error": "잘못된 사용자명 또는 비밀번호"}), 401
            
            token = create_session_token()
            expires_at = datetime.now() + timedelta(days=7)
            
            cursor.execute(
                "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
                (user['id'], token, expires_at)
            )
            conn.commit()
            
            logger.info(f"사용자 로그인: {username}")
            return jsonify({
                "message": "로그인 성공",
                "token": token,
                "user": {
                    "id": user['id'],
                    "username": user['username'],
                    "email": user['email']
                }
            })
            
    except Exception as e:
        logger.error(f"로그인 오류: {e}")
        return jsonify({"error": "로그인 처리 실패"}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
    return jsonify({"message": "로그아웃 완료"})

@app.route('/api/chat-sessions', methods=['GET'])
def get_chat_sessions():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    
    if not user:
        return jsonify({"error": "인증이 필요합니다"}), 401
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT cs.*,
                   (SELECT COUNT(*) FROM messages WHERE chat_session_id = cs.id) as message_count,
                   (SELECT question FROM messages WHERE chat_session_id = cs.id ORDER BY created_at ASC LIMIT 1) as first_question
            FROM chat_sessions cs
            WHERE cs.user_id = ?
            ORDER BY cs.updated_at DESC
        ''', (user['id'],))
        
        sessions = []
        for row in cursor.fetchall():
            title = row['title']
            if row['first_question']:
                title = row['first_question'][:40] + ("..." if len(row['first_question']) > 40 else "")
            
            sessions.append({
                "id": row['id'],
                "title": title,
                "message_count": row['message_count'] or 0,
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            })
        
        return jsonify({"sessions": sessions})

@app.route('/api/chat-sessions', methods=['POST'])
def create_chat_session():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    
    if not user:
        return jsonify({"error": "인증이 필요합니다"}), 401
    
    title = f"새 대화 {datetime.now().strftime('%m/%d %H:%M')}"
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
            (user['id'], title)
        )
        session_id = cursor.lastrowid
        conn.commit()
    
    return jsonify({"id": session_id, "title": title})

@app.route('/api/chat-sessions/<int:session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    
    if not user:
        return jsonify({"error": "인증이 필요합니다"}), 401
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session or session['user_id'] != user['id']:
            return jsonify({"error": "권한이 없습니다"}), 403
        
        cursor.execute("DELETE FROM messages WHERE chat_session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()
    
    return jsonify({"message": "삭제 완료"})

@app.route('/api/chat-sessions/<int:session_id>/messages', methods=['GET'])
def get_messages(session_id):
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    
    if not user:
        return jsonify({"error": "인증이 필요합니다"}), 401
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session or session['user_id'] != user['id']:
            return jsonify({"error": "권한이 없습니다"}), 403
        
        cursor.execute('''
            SELECT id, question, answer, sources, created_at,
                   CASE 
                       WHEN answer_type IS NULL THEN 'combined'
                       ELSE answer_type 
                   END as answer_type
            FROM messages
            WHERE chat_session_id = ?
            ORDER BY created_at ASC
        ''', (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            try:
                sources = json.loads(row['sources']) if row['sources'] else []
            except:
                sources = []
                
            messages.append({
                "id": row['id'],
                "question": row['question'],
                "answer": row['answer'],
                "sources": sources,
                "answer_type": row['answer_type'],
                "created_at": row['created_at']
            })
        
        return jsonify({"messages": messages})

@app.route('/api/chat', methods=['POST'])
def chat():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    
    if not user:
        return jsonify({"error": "인증이 필요합니다"}), 401
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        session_id = data.get('session_id')
        
        if not question:
            return jsonify({"error": "질문을 입력해주세요"}), 400
        
        if not session_id:
            return jsonify({"error": "세션 ID가 필요합니다"}), 400
        
        logger.info(f"질문 수신 [{user['username']}]: {question}")
        
        if not rag_system["client"]:
            return jsonify({"error": "AI 서비스가 설정되지 않았습니다"}), 500
        
        try:
            rag_context = ""
            sources = []
            
            if rag_system["loaded"]:
                chunks = search_chunks(question, k=3)
                if chunks:
                    rag_context_parts = []
                    for chunk in chunks:
                        header = f"[{chunk['source']}, 단락 {chunk['chunk_id']}"
                        if chunk.get("article_numbers"):
                            header += f", 조문: {', '.join(chunk['article_numbers'])}"
                        if chunk.get("isms_controls"):
                            header += f", ISMS: {', '.join(chunk['isms_controls'])}"
                        header += "]"
                        rag_context_parts.append(f"{header}\n{chunk['text']}")
                    rag_context = "\n\n".join(rag_context_parts)
                    sources = [
                        {
                            "source": chunk["source"],
                            "chunk_id": chunk["chunk_id"],
                            "article_numbers": chunk.get("article_numbers", []),
                            "isms_controls": chunk.get("isms_controls", [])
                        } for chunk in chunks
                    ]
                    logger.info(f"RAG 검색 결과: {len(chunks)}개 청크")
                else:
                    logger.info("RAG 검색 결과 없음")
            
            answer = generate_combined_answer(question, rag_context)
            answer_type = "combined" if rag_context else "general"
            logger.info(f"답변 생성 완료 ({answer_type}): {len(answer)}자")
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            sources = []
            answer_type = "error"
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO messages (chat_session_id, user_id, question, answer, sources, answer_type)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (session_id, user['id'], question, answer, json.dumps(sources), answer_type)
            )
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            conn.commit()
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "metadata": {
                "answer_type": answer_type,
                "rag_used": bool(rag_context),
                "model": "gemini-2.5-flash",
                "web_search_enabled": True,
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"채팅 오류: {e}")
        return jsonify({"error": f"처리 실패: {str(e)}"}), 500

@app.route('/')
def home():
    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="3;url=https://jin182.github.io/AI-web-basic-front/">
        <title>CPPG & ISMS-P 학습 가이드 서비스</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0; padding: 0;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container { 
                max-width: 600px; 
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                text-align: center;
            }
            .status { 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
            }
            .success { 
                background: #d4edda; 
                color: #155724; 
                border: 1px solid #c3e6cb; 
            }
            .info { 
                background: #cce7ff; 
                color: #004085; 
                border: 1px solid #b8daff; 
            }
            .btn { 
                display: inline-block; 
                padding: 12px 24px; 
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white; 
                text-decoration: none; 
                border-radius: 8px; 
                margin: 10px;
                font-weight: 600;
                transition: transform 0.2s;
            }
            .btn:hover { transform: translateY(-2px); }
            .loading {
                display: inline-block;
                width: 20px; height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #1e3c72;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CPPG & ISMS-P 학습 가이드</h1>
            
            <div class="status success">
                <h3>백엔드 엔진 로드 완료</h3>
                <p>Gemini 2.5 Flash 고성능 분산 인젝션<br>
                <small>법령 가이드 기반 RAG 컨텍스트 + 실시간 보안 웹 검색 동기화 활성화</small></p>
            </div>
            
            <div class="status info">
                <h3><span class="loading"></span> 프론트엔드 포트 브릿지 매핑 중...</h3>
                <p>3초 후 자동으로 프론트엔드 대시보드(https://jin182.github.io/AI-web-basic-front/)로 라우팅됩니다.</p>
            </div>
            
            <div style="margin-top: 30px;">
                <a href="https://jin182.github.io/AI-web-basic-front/" class="btn">즉시 대시보드 이동</a>
                <a href="/api/health" class="btn">엔진 메트릭 상태(Health)</a>
            </div>
        </div>
        <script>
            setTimeout(() => {
                window.location.href = 'https://jin182.github.io/AI-web-basic-front/';
            }, 3000);
            fetch('https://jin182.github.io/AI-web-basic-front/')
                .then(() => { window.location.href = 'https://jin182.github.io/AI-web-basic-front/'; })
                .catch(() => { console.log('프론트엔드가 실행되지 않았습니다'); });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "rag_available": rag_system["available"],
        "rag_loaded": rag_system["loaded"],
        "gemini_configured": bool(rag_system["client"]),
        "model": "gemini-2.5-flash",
        "web_search_enabled": True,
        "mode": "hybrid" if rag_system["loaded"] and rag_system["client"] else "ai_only" if rag_system["client"] else "limited",
        "features": ["security_web_search", "isms_compliance_check", "cppg_exam_router"],
        "error": rag_system.get("error"),
        "timestamp": time.time()
    })

# ======================== 초기화 ========================

def reset_database():
    db_path = Path(DATABASE_PATH)
    if db_path.exists():
        logger.warning(f"기존 데이터베이스 파일 삭제: {db_path}")
        db_path.unlink()
    init_database()
    logger.info("데이터베이스 완전 초기화 완료")

def initialize():
    logger.info("보안 학습 가이드 백엔드 커널 엔진 초기화")
    
    try:
        init_database()
    except Exception as e:
        logger.error(f"데이터베이스 구조 예외 발생: {e}")
        reset_database()
    
    if RAG_AVAILABLE:
        if load_rag_system():
            logger.info("🚨 [하이브리드 모드 가동]: 로컬 TF-IDF RAG 인덱스 + Gemini 2.5 + 실시간 웹 검색")
            return
            
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        rag_system["client"] = genai.Client(api_key=api_key)
        logger.info("⚠️ [AI 전용 가동]: 로컬 가이드 인덱스 누락으로 외부 KISA/개인정보위 실시간 웹 검색 및 일반 컨설팅 지식으로 구동")
    else:
        logger.error("❌ [치명적 오류] GEMINI_API_KEY 환경 변수가 식별되지 않아 AI 어시스턴트 기능이 중단되었습니다.")

if __name__ == '__main__':
    try:
        initialize()
        port = int(os.environ.get('PORT', 8000))
        logger.info(f"서버 정상 빌딩: https://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"서버 런타임 크래시: {e}")
        exit(1)