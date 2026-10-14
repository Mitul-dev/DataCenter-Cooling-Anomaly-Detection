$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (Test-Path -LiteralPath "$taskRoot\.venv\Scripts\python.exe") {
    & "$taskRoot\.venv\Scripts\python.exe" -m streamlit run "$taskRoot\dashboard\app.py"
} else {
    python -m streamlit run "$taskRoot\dashboard\app.py"
}
