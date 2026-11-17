# Cria o ambiente virtual e instala as dependências fixadas (Windows / PowerShell).
# Uso (na raiz do repositório):  .\scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -c "import src, evidently, mlflow, pandera, sklearn; print('Ambiente OK')"
