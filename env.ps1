# Dot-source this file before running project commands.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TMP = Join-Path $ProjectRoot 'cache\tmp'
$env:TEMP = $env:TMP
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot 'cache\pip'
$env:HF_HOME = Join-Path $ProjectRoot 'cache\hf'
$env:HF_DATASETS_CACHE = Join-Path $ProjectRoot 'cache\datasets'
$env:TRANSFORMERS_CACHE = Join-Path $ProjectRoot 'cache\hf\transformers'
$env:TORCH_HOME = Join-Path $ProjectRoot 'cache\torch'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $ProjectRoot 'cache\torchinductor'
$env:PYTHONNOUSERSITE = '1'
