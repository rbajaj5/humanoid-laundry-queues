param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$segments = @(
    @{start=0; end=6; text="This humanoid has twenty nine joints, a free base, and per link masses and inertias."},
    @{start=6; end=15; text="Both robots receive the same reaching commands and synthetic delays. The simulation includes gravity, floor contact, joint torque limits, and body inertia."},
    @{start=15; end=24; text="On the left, joint groups apply commands as they arrive. Delayed leg updates can leave the arms and legs working on different command versions."},
    @{start=24; end=33; text="On the right, the supervisor waits for a complete whole body command. This preserves coordination, but makes the robot wait for the slowest group."},
    @{start=33; end=40; text="Across eight matched seeds, synchronization reduced peak body tilt. Both robots stayed upright and reached the target."},
    @{start=40; end=48; text="This is an exploratory standing and reaching test, not a trained laundry robot. Clothing, door manipulation, and hardware validation still remain."}
)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$segments | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot "output/narration.json") -Encoding utf8
try {
    for ($i=0; $i -lt $segments.Count; $i++) {
        $path = Join-Path $PSScriptRoot "output/narration_$i.wav"
        $synth.SetOutputToWaveFile($path)
        $synth.Speak($segments[$i].text)
        $synth.SetOutputToNull()
    }
} finally { $synth.Dispose() }
& $Python (Join-Path $PSScriptRoot "assemble_audio.py")
if ($LASTEXITCODE -ne 0) { throw "Audio assembly failed" }
