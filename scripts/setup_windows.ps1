[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [switch]$Recreate,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Get-PythonVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    $Version = & $Executable @PrefixArguments -c `
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to run Python: $Executable $PrefixArguments"
    }
    return [version]($Version.Trim())
}

function Assert-SupportedPython {
    param([Parameter(Mandatory = $true)][version]$Version)

    if ($Version -lt [version]"3.11" -or $Version -ge [version]"3.13") {
        throw "Python $Version is unsupported. Install Python 3.11 (recommended) or use the Conda instructions in README.md."
    }
}

Set-Location -LiteralPath $ProjectRoot

if ($PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found: $PythonPath"
    }
    $BasePython = (Resolve-Path -LiteralPath $PythonPath).Path
    $BaseArguments = @()
}
else {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $Launcher) {
        throw "The Windows py launcher was not found. Pass -PythonPath with a Python 3.11-3.12 executable, or use the Conda setup in README.md."
    }
    $BasePython = $Launcher.Source
    $BaseArguments = @("-3.11")
}

$RequestedVersion = Get-PythonVersion -Executable $BasePython -PrefixArguments $BaseArguments
Assert-SupportedPython -Version $RequestedVersion
Write-Host "Using Python $RequestedVersion from $BasePython"

if (Test-Path -LiteralPath $VenvPath) {
    $ExistingVersion = $null
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        try {
            $ExistingVersion = Get-PythonVersion -Executable $VenvPython
        }
        catch {
            $ExistingVersion = $null
        }
    }

    $NeedsRecreate = $null -eq $ExistingVersion -or `
        $ExistingVersion.Major -ne $RequestedVersion.Major -or `
        $ExistingVersion.Minor -ne $RequestedVersion.Minor

    if ($Recreate) {
        $ResolvedVenv = (Resolve-Path -LiteralPath $VenvPath).Path
        if ((Split-Path -Parent $ResolvedVenv) -ne $ProjectRoot) {
            throw "Refusing to remove a virtual environment outside the project: $ResolvedVenv"
        }
        Write-Host "Removing existing project virtual environment: $ResolvedVenv"
        Remove-Item -LiteralPath $ResolvedVenv -Recurse -Force
    }
    elseif ($NeedsRecreate) {
        $Detected = if ($ExistingVersion) { $ExistingVersion } else { "unreadable" }
        throw ".venv uses Python $Detected but Python $RequestedVersion was requested. Rerun with -Recreate; do not create a new venv over the existing directory."
    }
    else {
        Write-Host "Reusing compatible .venv with Python $ExistingVersion"
    }
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $BasePython @BaseArguments -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
}

$Extras = if ($Dev) { ".[dev,notebook,ui]" } else { ".[notebook,ui]" }
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $VenvPython -m pip install -e $Extras
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependency check failed." }
& (Join-Path $VenvPath "Scripts\rag-cuisine.exe") validate-corpus
if ($LASTEXITCODE -ne 0) { throw "Corpus validation failed." }

Write-Host ""
Write-Host "Environment ready. Activate it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then build the local index with:"
Write-Host "  rag-cuisine build-index"
Write-Host "And open the visual demo with:"
Write-Host "  jupyter lab notebooks\Interactive_Demo.ipynb"
