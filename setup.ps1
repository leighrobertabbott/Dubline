param(
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VendorRoot = Join-Path $ProjectRoot "vendor"
$ToolsRoot = Join-Path $VendorRoot "tools"
$Upstream = Join-Path $VendorRoot "index-tts"
$Runtime = Join-Path $Upstream ".venv\Scripts\python.exe"
$RuntimeRevision = "4f8792ff120cd3ea470dd511e997a17c86cddd10"
$UvVersion = "0.12.5"

function Remove-DublineDirectory {
    param([Parameter(Mandatory)][string]$Path)
    $ResolvedVendor = [IO.Path]::GetFullPath($VendorRoot).TrimEnd('\') + '\'
    $ResolvedTarget = [IO.Path]::GetFullPath($Path).TrimEnd('\') + '\'
    if (-not $ResolvedTarget.StartsWith($ResolvedVendor, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Setup refused to remove a folder outside Dubline's vendor directory."
    }
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Dubline requires a 64-bit version of Windows."
}
if ($env:PROCESSOR_ARCHITECTURE -notin @("AMD64", "x86_64")) {
    throw "This release supports 64-bit Intel/AMD Windows. Detected: $env:PROCESSOR_ARCHITECTURE"
}

New-Item -ItemType Directory -Force -Path $VendorRoot, $ToolsRoot | Out-Null
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($UvCommand) {
    $Uv = $UvCommand.Source
} else {
    $Uv = Join-Path $ToolsRoot "uv.exe"
    if (-not (Test-Path -LiteralPath $Uv)) {
        Write-Host "Preparing Dubline's setup helper..." -ForegroundColor Green
        $env:UV_UNMANAGED_INSTALL = $ToolsRoot
        Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
    }
}
if (-not (Test-Path -LiteralPath $Uv)) {
    $UvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $UvCommand) { throw "The setup helper could not be installed." }
    $Uv = $UvCommand.Source
}

Write-Host "Preparing Python 3.11..." -ForegroundColor Green
& $Uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 could not be prepared." }

if (-not (Test-Path -LiteralPath (Join-Path $Upstream "pyproject.toml"))) {
    Write-Host "Downloading the pinned IndexTTS runtime..." -ForegroundColor Green
    $Git = Get-Command git -ErrorAction SilentlyContinue
    if ($Git) {
        & $Git.Source clone --filter=blob:none --no-checkout https://github.com/index-tts/index-tts.git $Upstream
        if ($LASTEXITCODE -ne 0) { throw "The IndexTTS runtime could not be downloaded." }
        & $Git.Source -C $Upstream checkout --detach $RuntimeRevision
        if ($LASTEXITCODE -ne 0) { throw "The tested IndexTTS runtime revision could not be selected." }
    } else {
        $Archive = Join-Path $VendorRoot "index-tts-$RuntimeRevision.zip"
        $Expanded = Join-Path $VendorRoot "index-tts-download"
        Remove-DublineDirectory -Path $Expanded
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/index-tts/index-tts/archive/$RuntimeRevision.zip" -OutFile $Archive
        Expand-Archive -LiteralPath $Archive -DestinationPath $Expanded -Force
        $DownloadedRoot = Get-ChildItem -LiteralPath $Expanded -Directory | Select-Object -First 1
        if (-not $DownloadedRoot) { throw "The IndexTTS download had an unexpected layout." }
        Move-Item -LiteralPath $DownloadedRoot.FullName -Destination $Upstream
        Remove-Item -LiteralPath $Archive -Force
        Remove-DublineDirectory -Path $Expanded
    }
    Set-Content -LiteralPath (Join-Path $Upstream ".dubline-runtime-revision") -Value $RuntimeRevision -NoNewline
}

Write-Host "Installing the tested local CUDA runtime. This first step can take several minutes..." -ForegroundColor Green
& $Uv sync --project $Upstream --python 3.11 --frozen
if ($LASTEXITCODE -ne 0) { throw "The IndexTTS runtime could not be installed." }
& $Runtime (Join-Path $ProjectRoot "scripts\patch_index_tts.py") $Upstream
if ($LASTEXITCODE -ne 0) { throw "Dubline's tested laptop-safe IndexTTS adjustment could not be applied." }

Write-Host "Installing Dubline's local service..." -ForegroundColor Green
& $Uv pip install --python $Runtime -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dubline's service packages could not be installed." }
& $Uv pip install --python $Runtime "uv==$UvVersion" "hf-xet==1.6.0" "melband-roformer-infer==0.1.5" "numpy==2.2.6" "opencv-python==4.12.0.88"
if ($LASTEXITCODE -ne 0) { throw "Dubline's model download support could not be installed." }
& $Uv pip install --python $Runtime "llama-cpp-python==0.3.34" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "Dubline's translation runtime could not be installed." }
& $Runtime -c "import torch, llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'CUDA LLM offload is unavailable'; print('CUDA translation offload ready')"
if ($LASTEXITCODE -ne 0) { throw "Dubline's CUDA translation runtime did not pass its verification." }

& $Runtime -c "import fastapi, torch, uvicorn; print('Local service ready · PyTorch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "The local service did not pass its final import check." }

Write-Host "Bootstrap complete. The setup wizard will download and verify your models." -ForegroundColor Green
if ($Launch) {
    & (Join-Path $ProjectRoot "start-dubline.ps1")
}
