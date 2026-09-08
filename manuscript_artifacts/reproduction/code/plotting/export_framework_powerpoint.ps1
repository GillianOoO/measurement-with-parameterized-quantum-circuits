param([Parameter(Mandatory=$true)][string]$Source,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$outputPath = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw 'Output already exists' }
$existing = @(Get-Process POWERPNT -ErrorAction SilentlyContinue)
$application = $null
$presentation = $null
$previousSecurity = $null
try {
    $application = New-Object -ComObject PowerPoint.Application
    $previousSecurity = $application.AutomationSecurity
    $application.AutomationSecurity = 3
    # ReadOnly=true, Untitled=false, WithWindow=false. No source edits.
    $presentation = $application.Presentations.Open($sourcePath, -1, 0, 0)
    $presentation.SaveAs($outputPath, 32)
    Write-Output ('Exported with PowerPoint ' + $application.Version)
} finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($null -ne $application) {
        if ($null -ne $previousSecurity) { $application.AutomationSecurity = $previousSecurity }
        if ($existing.Count -eq 0) { $application.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($application)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
