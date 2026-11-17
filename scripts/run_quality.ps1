# Lint + formatação + testes com cobertura (mesmas verificações do GitHub Actions).
$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pytest --cov --cov-report=term-missing --cov-report=html
exit $LASTEXITCODE
