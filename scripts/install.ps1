Param(
    [string]$Requirements = "requirements.txt"
)

Write-Host "Installing dependencies from $Requirements" -ForegroundColor Cyan

$process = Start-Process -FilePath "python" -ArgumentList "-m", "pip", "install", "-r", $Requirements -NoNewWindow -PassThru -Wait

if ($process.ExitCode -eq 0) {
    Write-Host "\nDependencies installed successfully." -ForegroundColor Green
    Write-Host "Launching welcome banner..." -ForegroundColor Cyan
    python -m mobile_game_analytics_pipeline.welcome
} else {
    Write-Host "\npip install failed (exit code $($process.ExitCode)). Skipping banner." -ForegroundColor Red
    exit $process.ExitCode
}
