# README_WEB.md — VET-TINY-GPT web
1. Instalación: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   + `pip install -r requirements.txt` en `C:\VET-TINY-LLM-venv`.
2. Activación: `. .\env-fast.ps1` (fija venv C: y `PYTHONPATH`).
3. Inicio: `.\run_web.bat` o `streamlit run app/streamlit_app.py --server.port 8501`.
4. URL: http://localhost:8501
5. Selección de checkpoint: barra lateral (lista `checkpoints/**/*.pt`).
6. Chat: página Chat, con especie e historial. Generación: página Generación.
7. Controles: temperature, top_p, max_new_tokens, repetition_penalty (CPU: máx 150).
8. Métricas: página Evaluación (archivos reales de `outputs/`).
9. Arquitectura: página Arquitectura (real, indica lo ausente).
10. Historia clínica: formulario + ANALIZAR (apoyo educativo, sin diagnósticos).
