$ErrorActionPreference = 'Stop'
$taskTools = Join-Path $env:RUNNER_TEMP 'native-renderer'
New-Item -ItemType Directory -Force $taskTools | Out-Null
$taskMsi = Join-Path $taskTools 'LibreOffice.msi'
$taskExtract = Join-Path $taskTools 'extracted'
Invoke-WebRequest -Uri 'https://download.documentfoundation.org/libreoffice/stable/26.8.0/win/x86_64/LibreOffice_26.8.0_Win_x86-64.msi' -OutFile $taskMsi
if ((Get-FileHash $taskMsi -Algorithm SHA256).Hash -ne '4AA6C6E1895F4055104EFFCB556BD3362D20C6AD707C149543304F395EF9DB95') {
    throw 'LibreOffice installer checksum mismatch'
}
$taskProcess = Start-Process -FilePath msiexec.exe -ArgumentList @(
    '/a', ('"' + $taskMsi + '"'), '/qn', ('TARGETDIR="' + $taskExtract + '"')
) -WindowStyle Hidden -PassThru -Wait
if ($taskProcess.ExitCode -ne 0) { throw "LibreOffice extraction failed: $($taskProcess.ExitCode)" }
$taskProgram = Join-Path $taskExtract 'program'
if (-not (Test-Path -LiteralPath (Join-Path $taskProgram 'soffice.com'))) {
    throw 'LibreOffice executable is missing'
}
$taskProgram | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
