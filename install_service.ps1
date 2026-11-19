# Run in an ADMIN PowerShell from the project folder:
#   powershell -ExecutionPolicy Bypass -File install_service.ps1
#   (options)  -Config C:\path\config.yaml   -ExePath C:\path\gate_engine.exe   -AtLogon
# Registers a Task Scheduler task that starts the headless engine at boot and restarts it if it stops.
param(
  [string]$Root = $PSScriptRoot,
  [string]$Config = (Join-Path $PSScriptRoot "config.yaml"),
  [string]$ExePath = "",
  [switch]$AtLogon
)
$TaskName = "ForkliftGateEngine"
if ($ExePath -ne "") {
  $action = New-ScheduledTaskAction -Execute $ExePath -Argument "--config `"$Config`"" -WorkingDirectory $Root
} else {
  $py = Join-Path $Root ".venv\Scripts\pythonw.exe"      # pythonw = no console window
  if (-not (Test-Path $py)) { throw "Not found: $py  (run setup_client.bat first)" }
  $action = New-ScheduledTaskAction -Execute $py -Argument "gate_engine.py --config `"$Config`"" -WorkingDirectory $Root
}
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
if ($AtLogon) {
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
} else {
  $trigger = New-ScheduledTaskTrigger -AtStartup
  $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
}
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Write-Host "Registered task '$TaskName'. Start now:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Remove:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
