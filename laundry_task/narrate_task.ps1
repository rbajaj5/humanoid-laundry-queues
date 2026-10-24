param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$out = Join-Path $PSScriptRoot "output"
$segments = Get-Content -LiteralPath (Join-Path $out "task_narration.json") -Raw | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
try {
    for ($i=0; $i -lt $segments.Count; $i++) {
        $synth.SetOutputToWaveFile((Join-Path $out "task_narration_$i.wav"))
        $synth.Speak($segments[$i].text)
        $synth.SetOutputToNull()
    }
} finally { $synth.Dispose() }
& $Python -m laundry_task.assemble_task_audio
if ($LASTEXITCODE -ne 0) { throw "Task audio assembly failed" }
