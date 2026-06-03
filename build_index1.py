"""
CPPG / ISMS-P 특화 법률 및 인증 가이드라인 RAG 인덱스 빌드 스크립트 
(KISA 공식 문서 최적화 / OCR 불필요)
"""

import os, json, re, argparse
from pathlib import Path
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from joblib import dump
from typing import List, Dict, Tuple

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120
ALLOWED_EXTS = {".pdf"}

def clean_text(s: str) -> str:
    """CPPG/ISMS-P 및 표(Table) 구조 텍스트 최적화"""
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u200b", "", s)
    
    # 불필요한 따옴표 및 콤마 찌꺼기 제거
    s = re.sub(r'^,*"+|"+$|",*"', ' ', s, flags=re.MULTILINE) 
    
    # 법률 조문 정규화
    s = re.sub(r"제\s*(\d+)\s*조", r"제\1조", s)
    s = re.sub(r"제\s*(\d+)\s*조의\s*(\d+)", r"제\1조의\2", s)
    s = re.sub(r"제\s*(\d+)\s*항", r"제\1항", s)
    s = re.sub(r"제\s*(\d+)\s*호", r"제\1호", s)
    
    # CPPG 실무/보안 용어 정규화
    term_mapping = {
        r"개인\s*정보\s*처리자": "개인정보처리자",
        r"정보\s*통신\s*서비스\s*제공자": "정보통신서비스제공자",
        r"개인\s*정보\s*취급자": "개인정보취급자",
        r"개인\s*정보\s*보호\s*위원회": "개인정보보호위원회",
        r"안전성\s*확보\s*조치": "안전성확보조치",
        r"개인\s*정보": "개인정보",
        r"가명\s*정보": "가명정보",
        r"가명\s*처리": "가명처리",
        r"정보\s*주체": "정보주체",
        r"영상\s*정보\s*처리\s*기기": "영상정보처리기기",
        r"인증\s*기준": "인증기준",
        r"결함\s*사례": "결함사례"
    }
    for pattern, replacement in term_mapping.items():
        s = re.sub(pattern, replacement, s)
    
    # 공식 문서 페이지 번호 및 헤더/푸터 등 쓸데없는 공백 제거
    s = re.sub(r"^\s*\\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\s*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\n{3,}", "\n\n", s)
    
    return s.strip()

def extract_text_from_pdf(path: Path) -> str:
    """PDF 텍스트 추출 (순수 PyMuPDF만 사용)"""
    print(f"  📄 PDF 처리 중: {path.name}")
    
    doc = fitz.open(path.as_posix())
    parts = []
    
    # 공식문서는 OCR 없이도 매우 깨끗하게 추출됩니다.
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            parts.append(text)
    
    raw_text = "\n".join(parts)
    doc.close()
    
    cleaned_text = clean_text(raw_text)
    
    filename_lower = path.name.lower()
    found_keywords = []
    
    if 'isms' in filename_lower:
        keywords = ['인증기준', '관리체계', '통제항목']
        found_keywords = [kw for kw in keywords if kw in cleaned_text]
    elif 'pipa' in filename_lower or '개인정보' in filename_lower:
        keywords = ['개인정보처리자', '가명정보', '정보주체']
        found_keywords = [kw for kw in keywords if kw in cleaned_text]
    elif 'icna' in filename_lower or '망법' in filename_lower or '정보통신' in filename_lower:
        keywords = ['정보통신서비스제공자', '방송통신위원회']
        found_keywords = [kw for kw in keywords if kw in cleaned_text]

    if found_keywords:
        print(f"    📋 핵심 키워드 확인: {found_keywords}")
    else:
        print(f"    ⚠️ 핵심 키워드 미발견 (문서 상태 확인 요망)")
        
    return cleaned_text

def smart_chunk_text(text: str, filename: str, chunk_size: int, overlap: int) -> List[str]:
    """표 구조 및 조문을 인식하는 지능적 청킹"""
    
    if 'isms' in filename.lower():
        # KISA 공식 문서는 "1.1.1 관리체계 수립" 형태로 깔끔하게 떨어지므로 정규식이 완벽히 작동합니다.
        splits = re.split(r'\n(?=\s*\d+\.\s*[가-힣A-Za-z]+|\s*(?:항목\s*:\s*)?[1-3]\.\d+(?:\.\d+)?\b)', text)
        print(f"    🔍 ISMS-P 통제항목 및 대목차 정밀 청킹 시도")
    else:
        splits = re.split(r'\n(?=제\d+조(?:의\d+)?)', text)
        print(f"    ⚖️ 법률 조문 기반 청킹 적용")

    meaningful_parts = [part.strip() for part in splits if len(part.strip()) > 30]
    chunks = []
    
    if len(meaningful_parts) >= 5:
        current_chunk = ""
        for part in meaningful_parts:
            if len(part) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                sub_chunks = sliding_window_chunk(part, chunk_size, overlap)
                chunks.extend(sub_chunks)
            elif len(current_chunk) > 0 and len(current_chunk) + len(part) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = part + "\n\n"
            else:
                current_chunk += part + "\n\n"
                
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    else:
        print(f"    ⚠️ 패턴 매칭 부족 ({len(meaningful_parts)}개 발견): 전체 텍스트 대상 촘촘한 슬라이딩 분할 적용")
        chunks = sliding_window_chunk(text, chunk_size, overlap)
        
    return chunks

def sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size: return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:].strip())
            break
            
        chunk = text[start:end]
        cut_point = max(chunk.rfind('. '), chunk.rfind('.'), chunk.rfind('\n'))
        
        if cut_point > chunk_size * 0.7:
            actual_end = start + cut_point + 1
            chunks.append(text[start:actual_end].strip())
            start = max(start + 1, actual_end - overlap)
        else:
            chunks.append(chunk.strip())
            start = end - overlap
            
    return chunks

def build_corpus(files: List[str], chunk_size: int, overlap: int) -> Tuple[List[str], List[Dict]]:
    corpus, meta = [], []
    
    for f in files:
        filename = os.path.basename(f)
        txt = extract_text_from_pdf(Path(f))
        chunks = smart_chunk_text(txt, filename, chunk_size, overlap)
        valid_chunks = [c for c in chunks if len(c.strip()) >= 30]
        
        print(f"    ✅ {len(valid_chunks)}개 청크 생성 완료")
        
        for i, chunk in enumerate(valid_chunks):
            corpus.append(chunk)
            
            article_numbers = re.findall(r'제(\d+조(?:의\d+)?)', chunk)
            isms_controls = re.findall(r'\b([1-3]\.\d+\.\d+)\b', chunk)
            
            meta.append({
                "source": filename,
                "chunk_id": i,
                "char_len": len(chunk),
                "article_numbers": article_numbers,
                "isms_controls": isms_controls,
                "preview": chunk[:150] + "..." if len(chunk) > 150 else chunk
            })
            
    return corpus, meta

def create_optimized_vectorizer(min_df: int, max_df: float, ngram_max: int) -> TfidfVectorizer:
    minimal_stopwords = ['이', '그', '것', '수', '등', '및', '대하여', '관하여', '따라', '경우']
    
    custom_token_pattern = (
        r'(?u)\b\w+\b|'
        r'제\d+조(?:의\d+)?|제\d+항|제\d+호|'
        r'[1-3]\.\d+\.\d+|'
        r'개인정보처리자|정보통신서비스제공자|개인정보취급자|정보주체|가명정보|영상정보처리기기|'
        r'수탁자|위탁자|결함사례|인증기준|CISO|CPO|안전성확보조치'
    )
    
    return TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, min(ngram_max, 2)),
        sublinear_tf=True,
        stop_words=minimal_stopwords,
        lowercase=False,
        max_features=50000,
        token_pattern=custom_token_pattern
    )

def main():
    ap = argparse.ArgumentParser(description="CPPG / ISMS-P 맞춤형 RAG 인덱스 빌드")
    ap.add_argument("pdfs", nargs="+", help="인덱싱할 PDF 파일들 (PIPA, ICNA, ISMS 등)")
    ap.add_argument("--outdir", default="./cppg_index", help="인덱스 저장 폴더")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE, help="청크 크기")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="오버랩 크기")
    
    args = ap.parse_args()
    
    pdfs = [Path(f).as_posix() for f in args.pdfs if Path(f).exists() and Path(f).suffix.lower() in ALLOWED_EXTS]
    
    if not pdfs:
        print("❌ 처리할 유효한 PDF 파일이 없습니다")
        return
        
    print(f"🛡️ CPPG/ISMS-P 보안 인덱스 빌드 시작 (대상: {len(pdfs)}개 문서)")
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    corpus, meta = build_corpus(pdfs, args.chunk, args.overlap)
    
    print(f"\n🔧 TF-IDF 벡터화 중...")
    vectorizer = create_optimized_vectorizer(min_df=1, max_df=0.95, ngram_max=2)
    X = vectorizer.fit_transform(corpus)
    
    print(f"💾 인덱스 저장 중...")
    save_npz(outdir / "tfidf_matrix.npz", X)
    dump(vectorizer, outdir / "vectorizer.joblib")
    
    with open(outdir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            
    with open(outdir / "chunks.txt", "w", encoding="utf-8") as f:
        for chunk in corpus:
            f.write(chunk.replace("\n", "\\n") + "\n")
            
    vocab = vectorizer.vocabulary_
    
    print(f"\n✅ 빌드 완료!")
    print(f"📚 생성된 총 청크: {X.shape[0]:,}개")
    
    key_terms = ['개인정보처리자', '가명정보', '결함사례', '수탁자', '안전성확보조치']
    found_terms = [term for term in key_terms if term in vocab]
    print(f"🔍 보호법 주요 용어 학습 확인: {found_terms}")

if __name__ == "__main__":
    main()