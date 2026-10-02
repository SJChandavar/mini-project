import os
import sys
import psutil

def stop_streamlit():
    pid_file = os.path.join("outputs", "logs", "streamlit.pid")
    stopped = False

    # 1. Check PID file
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                for child in proc.children(recursive=True):
                    try:
                        child.terminate()
                    except Exception:
                        pass
                proc.terminate()
                print(f"[SUCCESS] Stopped Streamlit process (PID: {pid}).")
                stopped = True
            else:
                print(f"[INFO] Process PID {pid} listed in PID file is no longer active.")
        except Exception as e:
            print(f"[WARNING] Could not terminate process from PID file: {e}")
        finally:
            if os.path.exists(pid_file):
                try:
                    os.remove(pid_file)
                except Exception:
                    pass

    # 2. Check for active Streamlit processes matching app.py
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = " ".join(proc.info['cmdline'] or []).lower()
            if "streamlit" in cmd and "app.py" in cmd:
                p = psutil.Process(proc.info['pid'])
                for child in p.children(recursive=True):
                    try:
                        child.terminate()
                    except Exception:
                        pass
                p.terminate()
                print(f"[SUCCESS] Terminated Streamlit process (PID: {proc.info['pid']}).")
                stopped = True
        except Exception:
            pass

    if stopped:
        print("[INFO] Application shutdown complete.")
    else:
        print("[INFO] No active Streamlit instance found for this project.")

if __name__ == "__main__":
    stop_streamlit()
