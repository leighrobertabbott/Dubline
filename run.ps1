$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $ProjectRoot "vendor\index-tts\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Runtime)) {
    & (Join-Path $ProjectRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Runtime)) {
        throw "Dubline's local runtime could not be prepared."
    }
}
$BundledTools = Join-Path $ProjectRoot "vendor\ffmpeg\bin"
if (Test-Path -LiteralPath $BundledTools) {
    $env:PATH = "$BundledTools;$env:PATH"
}
Set-Location -LiteralPath $ProjectRoot
& $Runtime -m uvicorn app.main:app --host 127.0.0.1 --port 8000
