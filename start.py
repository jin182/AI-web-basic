"""
AI 법률 상담 서비스 통합 실행 스크립트 (라이브러리 설치 불필요 버전)
- 백엔드 서버 (Flask): 8000번 포트 고정
- 프론트엔드 서버 (정적 파일): 3000번 포트 고정
- .env 수동 로드 기능 포함 (python-dotenv 불필요)
"""

import os
import sys
import time
import socket
import subprocess
import threading
import webbrowser
from pathlib import Path

# 색상 출력을 위한 ANSI 코드
class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color):
    """색상이 있는 메시지 출력"""
    print(f"{color}{message}{Colors.END}")

def print_banner():
    """시작 배너 출력"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 🏛️  CPPG &ISMS-P 서비스                        ║
    ║                                                               ║
    ║    정보통신망법, 개인정보보호법 등 관련 법률 기반             ║
    ║    Gemini AI + RAG 기술을 활용한 법률 상담 시스템             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Colors.PURPLE + Colors.BOLD)

def check_port(host, port):
    """포트 사용 가능 여부 확인"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        return result != 0

def kill_port(port):
    """특정 포트를 사용 중인 프로세스 종료 시도"""
    try:
        if sys.platform == "win32":
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                        return True
        else:
            result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(['kill', '-9', pid], capture_output=True)
                return True
    except Exception as e:
        print_colored(f"포트 {port} 정리 중 오류: {e}", Colors.YELLOW)
    return False

def handle_port_conflict(port, service_name):
    """포트 충돌 처리"""
    if check_port('localhost', port):
        return True 
    
    print_colored(f"⚠️  포트 {port}가 이미 사용 중입니다 ({service_name})", Colors.YELLOW)
    print_colored("다음 옵션을 선택하세요:", Colors.CYAN)
    print_colored("1. 기존 프로세스 자동 종료 후 계속", Colors.CYAN)
    print_colored("2. 수동으로 종료 후 계속", Colors.CYAN)  
    print_colored("3. 종료", Colors.CYAN)
    
    while True:
        choice = input("선택 (1/2/3): ").strip()
        if choice == '1':
            print_colored(f"포트 {port}를 사용하는 프로세스를 종료합니다...", Colors.YELLOW)
            if kill_port(port):
                time.sleep(2)
                if check_port('localhost', port):
                    print_colored(f"✅ 포트 {port} 정리 완료", Colors.GREEN)
                    return True
                else:
                    print_colored(f"❌ 포트 {port} 정리 실패", Colors.RED)
                    return False
            else:
                print_colored(f"❌ 자동 종료 실패. 수동으로 종료해주세요", Colors.RED)
                return False
        elif choice == '2':
            print_colored("기존 프로세스를 수동으로 종료한 후 Enter를 누르세요...", Colors.YELLOW)
            input()
            if check_port('localhost', port):
                print_colored(f"✅ 포트 {port} 사용 가능", Colors.GREEN)
                return True
            else:
                print_colored(f"❌ 포트 {port}가 여전히 사용 중입니다", Colors.RED)
                continue
        elif choice == '3':
            print_colored("종료합니다.", Colors.YELLOW)
            return False
        else:
            print_colored("잘못된 선택입니다. 1, 2, 또는 3을 입력하세요.", Colors.RED)

def check_dependencies():
    """필수 의존성 확인"""
    print_colored("📋 의존성 확인 중...", Colors.CYAN)
    if sys.version_info < (3, 8):
        print_colored("❌ Python 3.8 이상이 필요합니다", Colors.RED)
        return False
    
    print_colored(f"✅ Python {sys.version.split()[0]}", Colors.GREEN)
    
    required_packages = ['flask', 'flask_cors']
    optional_packages = ['numpy', 'scipy', 'sklearn', 'joblib', 'google.generativeai']
    
    missing_required = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_colored(f"✅ {package}", Colors.GREEN)
        except ImportError:
            missing_required.append(package)
            print_colored(f"❌ {package} (필수)", Colors.RED)
            
    for package in optional_packages:
        try:
            __import__(package)
            print_colored(f"✅ {package}", Colors.GREEN)
        except ImportError:
            print_colored(f"⚠️  {package} (선택적 - RAG 기능)", Colors.YELLOW)
            
    if missing_required:
        print_colored("\n필수 패키지 설치:", Colors.RED)
        print_colored(f"pip install {' '.join(missing_required)}", Colors.YELLOW)
        return False
        
    return True

def load_env_manually():
    """라이브러리 없이 .env 파일을 수동으로 읽어 환경 변수에 등록"""
    env_path = Path(".env")
    if env_path.exists():
        print_colored("📄 .env 파일을 읽어옵니다...", Colors.BLUE)
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 주석이 아니고 '='이 포함된 줄만 파싱
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"') # 따옴표 제거
                    os.environ[key] = value
                    
def check_environment():
    """환경 설정 확인"""
    print_colored("🌍 환경 설정 확인 중...", Colors.CYAN)
    
    # 1. 수동으로 .env 로드 실행 (여기서 API 키를 강제로 집어넣습니다)
    load_env_manually()
    
    # 2. Gemini API 키 확인
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if gemini_key:
        print_colored("✅ GEMINI_API_KEY 설정 및 로드 완료", Colors.GREEN)
    else:
        print_colored("❌ GEMINI_API_KEY 없음 - .env 파일에 키를 작성해주세요!", Colors.RED)
    
    # 3. 인덱스 파일 확인
    possible_paths = [Path("./index"), Path("../index"), Path("index")]
    index_found = False
    
    for path in possible_paths:
        if path.exists() and (path / "tfidf_matrix.npz").exists():
            print_colored(f"✅ RAG 인덱스 발견: {path.absolute()}", Colors.GREEN)
            index_found = True
            break
            
    if not index_found:
        print_colored("⚠️  RAG 인덱스 파일 없음 - 기본 모드로 실행", Colors.YELLOW)
        
    return True

def run_backend(port=8000):
    """백엔드 서버 실행"""
    print_colored(f"🚀 백엔드 서버 시작: http://localhost:{port}", Colors.GREEN)
    
    backend_path = Path("backend/app.py")
    app_path = Path("app.py")
    original_dir = os.getcwd()
    
    if backend_path.exists():
        os.chdir("backend")
        cmd = [sys.executable, "app.py"]
    elif app_path.exists():
        cmd = [sys.executable, "app.py"]
    else:
        print_colored("❌ backend/app.py 또는 app.py를 찾을 수 없습니다", Colors.RED)
        return None
        
    # 부모 프로세스(이 스크립트)가 방금 수동으로 읽어들인 환경 변수를 자식 프로세스(Flask)에 그대로 전달
    env = os.environ.copy()
    env['PORT'] = str(port)
    env['FLASK_ENV'] = 'development'
    
    try:
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1
        )
        os.chdir(original_dir)
        
        def log_backend():
            for line in process.stdout:
                print_colored(f"[Backend] {line.strip()}", Colors.BLUE)
                
        threading.Thread(target=log_backend, daemon=True).start()
        return process
    except Exception as e:
        print_colored(f"❌ 백엔드 서버 시작 실패: {e}", Colors.RED)
        os.chdir(original_dir)
        return None

def run_frontend(port=3000, directory="."):
    """프론트엔드 서버 실행"""
    print_colored(f"🌐 프론트엔드 서버 시작: http://localhost:{port}", Colors.GREEN)
    try:
        cmd = [sys.executable, "-m", "http.server", str(port), "--directory", str(directory)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return process
    except Exception as e:
        print_colored(f"❌ 프론트엔드 서버 시작 실패: {e}", Colors.RED)
        return None

def wait_for_server(host, port, timeout=30):
    """서버가 시작될 때까지 대기"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not check_port(host, port):
            return True
        time.sleep(0.5)
    return False

def main():
    """메인 실행 함수"""
    print_banner()
    
    if not check_dependencies():
        sys.exit(1)
        
    # 여기서 환경변수 검사 및 수동 로드를 실행합니다.
    check_environment()
    
    backend_port = 8000
    frontend_port = 3000
    
    print_colored(f"📡 백엔드 포트: {backend_port} ", Colors.CYAN)
    print_colored(f"🌐 프론트엔드 포트: {frontend_port} ", Colors.CYAN)
    
    if not handle_port_conflict(backend_port, "백엔드"): sys.exit(1)
    if not handle_port_conflict(frontend_port, "프론트엔드"): sys.exit(1)
    
    frontend_paths = [Path("frontend/index.html"), Path("index.html")]
    frontend_dir = None
    
    for path in frontend_paths:
        if path.exists():
            frontend_dir = path.parent
            break
            
    if not frontend_dir:
        print_colored("❌ frontend/index.html 또는 index.html을 찾을 수 없습니다", Colors.RED)
        sys.exit(1)
        
    print_colored(f"📁 프론트엔드 디렉토리: {frontend_dir.absolute()}", Colors.CYAN)
    print_colored("\n🔄 서버들을 시작하는 중...", Colors.YELLOW)
    
    backend_process = run_backend(backend_port)
    if not backend_process: sys.exit(1)
    
    print_colored("⏳ 백엔드 서버 시작 대기 중...", Colors.YELLOW)
    if not wait_for_server('localhost', backend_port):
        print_colored("❌ 백엔드 서버 시작 타임아웃", Colors.RED)
        backend_process.terminate()
        sys.exit(1)
    print_colored("✅ 백엔드 서버 시작 완료", Colors.GREEN)
    
    frontend_process = run_frontend(frontend_port, frontend_dir)
    if not frontend_process:
        backend_process.terminate()
        sys.exit(1)
        
    print_colored("⏳ 프론트엔드 서버 시작 대기 중...", Colors.YELLOW)
    if not wait_for_server('localhost', frontend_port):
        print_colored("❌ 프론트엔드 서버 시작 타임아웃", Colors.RED)
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(1)
    print_colored("✅ 프론트엔드 서버 시작 완료", Colors.GREEN)
    
    print_colored("\n" + "="*60, Colors.GREEN + Colors.BOLD)
    print_colored("🎉 모든 서버가 성공적으로 시작되었습니다!", Colors.GREEN + Colors.BOLD)
    print_colored("="*60, Colors.GREEN + Colors.BOLD)
    
    print_colored(f"\n🌐 웹 애플리케이션: http://localhost:{frontend_port}", Colors.CYAN + Colors.BOLD)
    print_colored(f"🔧 백엔드 API: http://localhost:{backend_port}", Colors.CYAN)
    print_colored(f"📊 서버 상태: http://localhost:{backend_port}/api/health", Colors.CYAN)
    print_colored("\n💡 브라우저가 자동으로 열리지 않으면 위 URL을 직접 방문하세요", Colors.YELLOW)
    print_colored("⏹️  종료하려면 Ctrl+C를 누르세요", Colors.YELLOW)
    
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open(f"http://localhost:{frontend_port}")
        except: pass
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        while True:
            if backend_process.poll() is not None:
                print_colored("❌ 백엔드 서버가 중지되었습니다", Colors.RED)
                break
            if frontend_process.poll() is not None:
                print_colored("❌ 프론트엔드 서버가 중지되었습니다", Colors.RED)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print_colored("\n\n🛑 사용자가 중지를 요청했습니다", Colors.YELLOW)
    finally:
        print_colored("🔄 서버들을 종료하는 중...", Colors.YELLOW)
        if backend_process:
            backend_process.terminate()
            try: backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired: backend_process.kill()
        if frontend_process:
            frontend_process.terminate()
            try: frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired: frontend_process.kill()
        print_colored("✅ 모든 서버가 정상적으로 종료되었습니다", Colors.GREEN)
        print_colored("👋 CPPG &ISMS-P 서비스를 이용해 주셔서 감사합니다!", Colors.PURPLE + Colors.BOLD)

if __name__ == "__main__":
    main()