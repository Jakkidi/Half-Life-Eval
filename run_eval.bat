@echo off
setlocal
rem Constraint Half-Life: one-click runner for Windows.
rem Usage: double-click, or from a prompt:  run_eval.bat llama3.1:8b,qwen2.5:7b [preset]
cd /d "%~dp0"
chcp 65001 >nul

rem "python" can be the Microsoft Store stub, so test that it actually runs.
set "PY=python"
python --version >nul 2>nul || set "PY=py"
%PY% --version >nul 2>nul || (
  echo Python 3.10+ was not found. Install it from python.org and tick "Add to PATH".
  goto :end
)

where ollama >nul 2>nul || (
  echo Ollama was not found. Install it from ollama.com, then run this again.
  goto :end
)

echo.
echo Models available in Ollama:
ollama list
echo.

set "MODELS=%~1"
set "PRESET=%~2"
if "%PRESET%"=="" set "PRESET=quick"
if "%MODELS%"=="" (
  set /p "MODELS=Model(s) to test, comma-separated [Enter = first in the list]: "
)
if "%MODELS%"=="" set "MODELS=auto"

echo.
echo Running preset "%PRESET%" on: %MODELS%
echo Progress prints below. You can close this window and run it again later; it resumes.
echo.
%PY% -m chl run --models %MODELS% --preset %PRESET% --out results\run.jsonl
if errorlevel 1 goto :end

echo.
echo Done. Opening the report...
start "" "results\run.html"

:end
echo.
pause
