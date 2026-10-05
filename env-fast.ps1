# Activa el venv rápido (C: NVMe). D: es USB lento: no instalar nada pesado en .venv.
$VenvPy = "C:\VET-TINY-LLM-venv\Scripts\python.exe"
$env:TMP = "C:\Users\alexi\AppData\Local\Temp\opencode"
$env:TEMP = $env:TMP
$env:PIP_CACHE_DIR = "C:\Users\alexi\AppData\Local\Temp\opencode\pip"
$env:HF_HOME = "C:\Users\alexi\AppData\Local\Temp\opencode\hf"
$env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets"
$env:TORCH_HOME = "C:\Users\alexi\AppData\Local\Temp\opencode\torch"
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = "D:\IA\VET-TINY-LLM"
$env:TIKTOKEN_CACHE_DIR = "C:\Users\alexi\AppData\Local\Temp\opencode\tiktoken"
Set-Alias -Name vpy -Value $VenvPy -Scope Global
