"""Vercel entrypoint for VET-TINY-GPT Streamlit app."""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the streamlit app
from app.streamlit_app import *

# Vercel expects the Streamlit app to be runnable
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", str(ROOT / "app" / "streamlit_app.py")]
    sys.exit(stcli.main())