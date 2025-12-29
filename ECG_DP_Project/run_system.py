
import subprocess
import time
import webbrowser
import os
import signal
import sys

def run_backend():
    """Run Flask backend"""
    print("Starting Flask backend...")
    backend_process = subprocess.Popen(
        [sys.executable, "backend.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for backend to start
    time.sleep(3)
    return backend_process

def run_frontend():
    """Open frontend in browser"""
    print("Opening frontend in browser...")
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    # Open in default browser
    webbrowser.open(f"file://{frontend_path}")

    print(f"Frontend available at: file://{frontend_path}")

def main():
    print("="*60)
    print("ECG DP Analysis System - Complete Package")
    print("="*60)
    print()
    print("Starting the system...")
    print()

    try:
        # Run backend
        backend = run_backend()

        # Run frontend
        run_frontend()

        print()
        print("="*60)
        print("System is running!")
        print("Backend: http://localhost:5000")
        print("Frontend: Open in browser")
        print()
        print("Press Ctrl+C to stop the system")
        print("="*60)
        print()

        # Keep running
        try:
            backend.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            backend.terminate()
            backend.wait()
            print("System stopped.")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
