[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'

# Always run from repo root (this script lives in scripts/)
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Virtualenv python not found at: $python. Create venv: python -m venv .venv"
}

& $python -m pytest @Args
exit $LASTEXITCODE
