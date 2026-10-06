param(
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [int]$TrainerProcessId = 0,
    [int]$TotalEpochs = 50,
    [int]$PollSeconds = 5
)

$run = (Resolve-Path -LiteralPath $RunRoot).Path
$log = Join-Path $run 'console.log'
$config = Join-Path $run 'config.json'
$started = if (Test-Path -LiteralPath $config) { (Get-Item -LiteralPath $config).LastWriteTime } else { Get-Date }
$lastDone = 0
$lastObserved = $started
$secondsPerBatch = $null

while ($true) {
    $lines = if (Test-Path -LiteralPath $log) { @(Get-Content -LiteralPath $log -Tail 80 -ErrorAction SilentlyContinue) } else { @() }
    $progressLine = $lines | Where-Object { $_ -match '^epoch \d+ batch \d+/\d+ loss ' } | Select-Object -Last 1
    $summaryLine = $lines | Where-Object { $_ -match '^Epoch \d+/\d+ \| train ' } | Select-Object -Last 1

    $epoch = 1
    $batch = 0
    $batchCount = 0
    $loss = $null
    if ($progressLine -match '^epoch (\d+) batch (\d+)/(\d+) loss ([0-9.]+)') {
        $epoch = [int]$Matches[1]
        $batch = [int]$Matches[2]
        $batchCount = [int]$Matches[3]
        $loss = $Matches[4]
    }
    if ($summaryLine -match '^Epoch (\d+)/(\d+) \| train ' -and [int]$Matches[1] -ge $epoch) {
        $epoch = [int]$Matches[1] + 1
        $batch = 0
    }

    if ($batchCount -gt 0) {
        $done = ($epoch - 1) * $batchCount + $batch
        $now = Get-Date
        if ($done -gt $lastDone) {
            if ($lastDone -gt 0) {
                $rate = ($now - $lastObserved).TotalSeconds / ($done - $lastDone)
                if ($rate -gt 0) {
                    $secondsPerBatch = if ($null -eq $secondsPerBatch) { $rate } else { 0.7 * $secondsPerBatch + 0.3 * $rate }
                }
            }
            $lastDone = $done
            $lastObserved = $now
        }
        $remaining = if ($null -eq $secondsPerBatch) { -1 } else { [int][math]::Max(0, ($TotalEpochs * $batchCount - $done) * $secondsPerBatch) }
        $estimate = if ($null -eq $secondsPerBatch) { $batch } else { [math]::Min($batchCount - 1, $batch + [int](($now - $lastObserved).TotalSeconds / $secondsPerBatch)) }
        $status = "Epoch $epoch/$TotalEpochs | logged batch $batch/$batchCount, estimated $estimate | loss $loss | approximate training ETA "
        $status += if ($remaining -ge 0) { [TimeSpan]::FromSeconds($remaining).ToString('d\.hh\:mm\:ss') } else { 'calculating' }
        Write-Progress -Activity 'Stage 1 radar-to-LiDAR training' -Status $status -PercentComplete ([math]::Min(100, 100 * $estimate / $batchCount)) -SecondsRemaining $remaining
    } else {
        Write-Progress -Activity 'Stage 1 radar-to-LiDAR training' -Status 'Waiting for the first batch-100 update; the trainer is still working.' -PercentComplete 0
    }

    if ($TrainerProcessId -gt 0 -and -not (Get-Process -Id $TrainerProcessId -ErrorAction SilentlyContinue)) {
        Write-Progress -Activity 'Stage 1 radar-to-LiDAR training' -Completed
        Write-Output 'Trainer process ended. Latest console output:'
        if ($lines.Count) { $lines | Select-Object -Last 12 }
        break
    }
    if ($epoch -gt $TotalEpochs) {
        Write-Progress -Activity 'Stage 1 radar-to-LiDAR training' -Completed
        Write-Output "All $TotalEpochs epochs finished. Results: $run"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
