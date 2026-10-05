@echo off
cd /d D:\VET-TINY-LLM
call C:\VET-TINY-LLM-venv\Scripts\activate.bat
set PYTHONPATH=D:\VET-TINY-LLM
streamlit run app/streamlit_app.py --server.port 8501
