import os
import sys
import subprocess
import socket

def start_app():
    # 1. Determine project directory (parent of tools directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_dir)

    # 2. Check app.py existence
    app_py = os.path.join(project_dir, "app.py")
    if not os.path.exists(app_py):
        print(f"[ERROR] Application file app.py not found at: {app_py}")
        sys.exit(1)

    # 3. Check port 8501 availability
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    res = s.connect_ex(('127.0.0.1', 8501))
    s.close()

    if res == 0:
        print("[ERROR] Port 8501 is already in use by another application or process.")
        print("Please stop the process running on port 8501 before starting the app.")
        sys.exit(2)

    # 4. Ensure outputs/logs folder exists and write PID
    logs_dir = os.path.join(project_dir, "outputs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    pid_file = os.path.join(logs_dir, "streamlit.pid")

    # 5. Launch Streamlit process
    python_exe = sys.executable
    cmd = [python_exe, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]

    print(f"Starting Streamlit application on http://localhost:8501 ...")
    p = subprocess.Popen(cmd, cwd=project_dir)

    # Write PID to file
    with open(pid_file, "w") as f:
        f.write(str(p.pid))

    print(f"[INFO] Streamlit server started (PID: {p.pid}).")
    return_code = p.wait()

    # Cleanup PID file when process finishes
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
        except Exception:
            pass

    sys.exit(return_code)

if __name__ == "__main__":
    start_app()
