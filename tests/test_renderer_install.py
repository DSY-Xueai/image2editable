import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_renderer_admin_image_does_not_overwrite_source_installer(tmp_path):
    shell = shutil.which('pwsh')
    if shell is None:
        pytest.skip('PowerShell is required')
    script = Path(__file__).resolve().parents[1] / 'scripts/install_release_renderer.ps1'
    command = r'''
function Invoke-WebRequest {
    param($Uri, $OutFile)
    New-Item -ItemType File -Path $OutFile | Out-Null
}
function Get-FileHash {
    param($Path, $Algorithm)
    return @{ Hash = '4AA6C6E1895F4055104EFFCB556BD3362D20C6AD707C149543304F395EF9DB95' }
}
function Start-Process {
    param($FilePath, $ArgumentList, $WindowStyle, [switch]$PassThru, [switch]$Wait)
    $source = [IO.Path]::GetFullPath($ArgumentList[1].Trim('"'))
    $target = [IO.Path]::GetFullPath(($ArgumentList | Where-Object { $_ -like 'TARGETDIR=*' }).Substring(10).Trim('"'))
    if ([IO.Path]::GetDirectoryName($source) -eq $target) {
        throw 'Administrative image would overwrite its source MSI'
    }
    if (-not (Test-Path -LiteralPath $source)) { throw 'Missing source installer' }
    $program = Join-Path $target 'program'
    New-Item -ItemType Directory -Force $program | Out-Null
    New-Item -ItemType File (Join-Path $program 'soffice.com') | Out-Null
    return @{ ExitCode = 0 }
}
& $env:RENDERER_INSTALL_SCRIPT
'''
    result = subprocess.run(
        [shell, '-NoProfile', '-NonInteractive', '-Command', command],
        env={**os.environ, 'RUNNER_TEMP': str(tmp_path),
             'GITHUB_PATH': str(tmp_path / 'github-path.txt'),
             'RENDERER_INSTALL_SCRIPT': str(script)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    program = Path((tmp_path / 'github-path.txt').read_text(encoding='utf-8-sig').strip())
    assert (program / 'soffice.com').is_file()
