#!/usr/bin/env python3
"""
Launch the Streamlit Web App for Handwriting Emotion Detection.

Run this script to start the interactive web interface:
    python run_app.py

Or use Streamlit directly:
    streamlit run app.py
"""
import subprocess
import sys
import webbrowser
import time

def main():
    """Launch the Streamlit app."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " LAUNCHING HANDWRITING EMOTION DETECTION WEB APP ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    print("🚀 Starting Streamlit server...")
    print()
    print("📌 The app will open in your browser at: http://localhost:8501")
    print()
    print("💡 Tips:")
    print("   • Upload .svc files or select from dataset")
    print("   • Click buttons to analyze handwriting")
    print("   • View trajectories and extract features")
    print("   • Press Ctrl+C to stop the server")
    print()
    print("─" * 80)
    print()

    try:
        # Start Streamlit app
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            cwd=".",
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Web app stopped.")
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
