import http.server
import socketserver
import webbrowser
from pathlib import Path

def main():
    PORT = 3000
    
    print("🌐 프론트엔드 서버")
    print("=" * 30)
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
        
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"✅ 서버 시작: http://localhost:{PORT}")
            
            try:
                webbrowser.open(f'http://localhost:{PORT}')
            except:
                pass
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 서버 종료")
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == '__main__':
    main()