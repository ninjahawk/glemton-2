# run_train.ps1 — resume-loop training launcher for glemton-2 (nano or base).
#
# Runs `python -m ton2.train <config>`, auto-resuming from the latest checkpoint
# on any crash, until final.pt appears or a safety sentinel halts it. Hardware
# safety is enforced inside train.py (ThermalGuard) regardless of how this is
# launched. Designed to run headless via Windows Task Scheduler (detached,
# survives terminal/agent close) or in the background.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_train.ps1 `
#       -Config configs\ton2-base.yaml -Name ton2-base
param(
    [string]$Config = "configs\ton2-base.yaml",
    [string]$Name = "ton2-base"
)
$ErrorActionPreference = "Continue"
$root = "C:\Users\jedin\Desktop\glemton-2"
Set-Location $root

# Detach hardening: stop the Fortran runtime (via numpy/MKL) aborting on console
# events (this killed a Glemton-1 relaunch). Inherited by child processes.
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

$py = ".venv\Scripts\python.exe"
$ckptDir = "checkpoints\$Name"
$log = "logs\$Name.log"
$sentinels = @("THERMAL_STOP", "GATE_FAIL", "DISK_LOW")
New-Item -ItemType Directory -Force -Path $ckptDir, "logs" | Out-Null

function Log($m) {
    $l = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Name] $m"
    Write-Host $l
    Add-Content "logs\$Name.run.log" $l -ErrorAction SilentlyContinue
}
function HasSentinel { foreach ($s in $sentinels) { if (Test-Path (Join-Path $ckptDir $s)) { return $s } } return $null }

Log "launcher up (pid $PID) config=$Config"

# Background watcher: prune checkpoints (keep newest 4 + every 500M-token milestone).
$watcher = Start-Job -ScriptBlock {
    param($root, $Name)
    $ckptDir = Join-Path $root "checkpoints\$Name"
    while ($true) {
        Start-Sleep -Seconds 600
        try {
            $ck = Get-ChildItem (Join-Path $ckptDir "step_*.pt") -ErrorAction SilentlyContinue
            $info = foreach ($c in $ck) { if ($c.Name -match "_tokens_(\d+)M") { [pscustomobject]@{ F = $c; T = [int]$Matches[1] } } }
            $sorted = $info | Sort-Object T
            $keep = ($sorted | Select-Object -Last 4).F.FullName
            foreach ($x in $sorted) {
                $mile = ($x.T % 500) -eq 0
                $new = $keep -contains $x.F.FullName
                $age = ((Get-Date) - $x.F.LastWriteTime).TotalMinutes
                if (-not $mile -and -not $new -and $age -gt 5) { Remove-Item $x.F.FullName -Force -ErrorAction SilentlyContinue }
            }
        } catch {}
    }
} -ArgumentList $root, $Name

$attempt = 0
while ($true) {
    if (Test-Path (Join-Path $ckptDir "final.pt")) { Log "final.pt present — done."; break }
    $sen = HasSentinel; if ($sen) { Log "safety sentinel [$sen] present — halting for review."; break }
    $attempt++
    $latest = Get-ChildItem (Join-Path $ckptDir "step_*.pt") -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($latest) { Log "attempt $attempt — resume $($latest.Name)"; & $py -m ton2.train $Config --resume $latest.FullName 2>&1 | Tee-Object -FilePath $log -Append }
    else { Log "attempt $attempt — fresh start"; & $py -m ton2.train $Config 2>&1 | Tee-Object -FilePath $log -Append }
    Log "train exited (code $LASTEXITCODE)"
    if (Test-Path (Join-Path $ckptDir "final.pt")) { break }
    $sen = HasSentinel; if ($sen) { Log "safety sentinel [$sen] — halting."; break }
    if ($attempt -ge 50) { Log "50 attempts — giving up to avoid a crash loop."; break }
    Start-Sleep -Seconds 30
}
Stop-Job $watcher -ErrorAction SilentlyContinue
Remove-Job $watcher -Force -ErrorAction SilentlyContinue
Log "run loop finished."
