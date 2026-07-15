param(
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$project = Join-Path $repoRoot 'apps\tippi-windows\Tippi.Windows.csproj'

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot 'artifacts\Tippi-win-x64'
}

$localDotnet = Join-Path $repoRoot '.tools\dotnet\dotnet.exe'
if (Test-Path $localDotnet) {
    $dotnet = $localDotnet
} else {
    $dotnet = (Get-Command dotnet -ErrorAction Stop).Source
}

& $dotnet publish $project `
    --configuration Release `
    --runtime win-x64 `
    --self-contained true `
    --output $OutputDirectory `
    -p:PublishSingleFile=false `
    -p:DebugType=None `
    -p:DebugSymbols=false

if ($LASTEXITCODE -ne 0) {
    throw "Tippi publish failed with exit code $LASTEXITCODE."
}

$exe = Join-Path $OutputDirectory 'Tippi.exe'
if (!(Test-Path $exe)) {
    throw "Publish completed but Tippi.exe was not found at $exe."
}

Write-Host "Tippi for Windows is ready: $exe"
