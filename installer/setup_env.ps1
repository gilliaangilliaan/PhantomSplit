param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet("auto", "cu121", "cu128", "cpu")][string]$Torch = "auto",
    [switch]$DetectOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Builds = @{
    cu121 = @{ Version = "2.5.1"; Label = "NVIDIA (CUDA 12.1)" }
    cu128 = @{ Version = "2.7.1"; Label = "NVIDIA RTX 50 series (CUDA 12.8)" }
    cpu   = @{ Version = "2.5.1"; Label = "CPU only" }
}

function Invoke-Tool($Exe, $Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$(Split-Path -Leaf $Exe) $($Arguments[0..1] -join ' ') failed (exit code $LASTEXITCODE)" }
}

try {
    $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    Write-Host "graphics cards found:"
    $gpus | ForEach-Object { Write-Host "  $_" }

    if ($Torch -eq "auto") {
        $nvidia = @($gpus | Where-Object { $_ -match "NVIDIA" })
        if ($nvidia | Where-Object { $_ -match "RTX\s*50\d0" }) { $Torch = "cu128" }
        elseif ($nvidia) { $Torch = "cu121" }
        else { $Torch = "cpu" }
    }
    $build = $Builds[$Torch]
    Write-Host "PyTorch build: $($build.Label) - torch $($build.Version)" -ForegroundColor Cyan
    if ($Torch -eq "cpu") {
        Write-Host "no NVIDIA card found, PhantomSplit will run on the CPU (much slower)" -ForegroundColor Yellow
    }
    if ($DetectOnly) { exit 0 }

    $tools = Join-Path $AppDir "tools"
    $uv = Join-Path $tools "uv.exe"
    $venv = Join-Path $AppDir ".venv"
    $python = Join-Path $venv "Scripts\python.exe"
    $requirements = Join-Path $PSScriptRoot "requirements.txt"
    $marker = Join-Path $venv "phantomsplit_env.txt"
    $wanted = "$($build.Version)+$Torch $((Get-FileHash $requirements -Algorithm SHA256).Hash)"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $AppDir "python"
    $env:UV_CACHE_DIR = Join-Path $AppDir ".uv-cache"

    $installedTorch = ""
    if (Test-Path $python) {
        $ErrorActionPreference = "Continue"
        try { $installedTorch = "$(& $python -c 'import torch; print(torch.__version__)' 2>$null)".Trim() } catch { $installedTorch = "" }
        $ErrorActionPreference = "Stop"
    }
    $torchReady = $installedTorch -eq "$($build.Version)+$Torch"
    $packagesReady = $torchReady -and (Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $wanted)

    if ($packagesReady) {
        Write-Host "`nthe python environment is already up to date, nothing to download" -ForegroundColor Green
        exit 0
    }

    if (-not (Test-Path $uv)) {
        Write-Host "`ndownloading uv (python installer)..."
        New-Item -ItemType Directory -Force $tools | Out-Null
        $zip = Join-Path $tools "uv.zip"
        Invoke-WebRequest "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
        Expand-Archive $zip -DestinationPath $tools -Force
        Remove-Item $zip
    }

    if ($torchReady) {
        Write-Host "`nPyTorch $installedTorch is already installed, keeping it" -ForegroundColor Green
    }
    else {
        Write-Host "`ncreating the python 3.10 environment..."
        if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
        Invoke-Tool $uv @("venv", "--python", "3.10", $venv)

        Write-Host "`ninstalling PyTorch (the NVIDIA build is about 2.5 GB, this takes a while)..."
        Invoke-Tool $uv @("pip", "install", "--python", $python, "torch==$($build.Version)", "--index-url", "https://download.pytorch.org/whl/$Torch")
    }

    Write-Host "`ninstalling the other packages..."
    Invoke-Tool $uv @("pip", "install", "--python", $python, "-r", $requirements)

    Write-Host "`nchecking the install..."
    Invoke-Tool $python @("-c", "import torch, soundfile, yaml, scipy, tqdm, einops, rotary_embedding_torch, beartype, tkinter; print('torch', torch.__version__, '| gpu available:', torch.cuda.is_available())")

    Set-Content -Path $marker -Value $wanted -Encoding ascii
    if (Test-Path $env:UV_CACHE_DIR) { Remove-Item -Recurse -Force $env:UV_CACHE_DIR }
    Write-Host "`nPhantomSplit environment is ready" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "`n[error] $_" -ForegroundColor Red
    Write-Host "check your internet connection, then run the installer again"
    Read-Host "press Enter to close"
    exit 1
}
