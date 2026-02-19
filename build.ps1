# build.ps1 - PowerShell build script for Epistemic Deconstructor Claude Skill
# Usage: .\build.ps1 [command]
# Commands: build, build-combined, package, validate, clean, list, help

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$SkillName = "epistemic-deconstructor"
$Version = "6.6.0"
$BuildDir = "build"
$DistDir = "dist"

function Show-Help {
    Write-Host "Epistemic Deconstructor Skill - Build Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build           - Build skill package structure"
    Write-Host "  build-combined  - Build single-file skill with inlined references"
    Write-Host "  package         - Create zip package"
    Write-Host "  package-combined - Create single-file skill in dist/"
    Write-Host "  package-tar     - Create tarball package"
    Write-Host "  validate        - Validate skill structure"
    Write-Host "  lint            - Check Python syntax"
    Write-Host "  test            - Run lint + unit tests + smoke tests"
    Write-Host "  clean           - Remove build artifacts"
    Write-Host "  list            - Show package contents"
    Write-Host "  install         - Show install instructions"
    Write-Host "  sync-skill      - Sync skill to ~/.claude/skills/"
    Write-Host "  help            - Show this help"
    Write-Host ""
    Write-Host "Skill: $SkillName v$Version" -ForegroundColor Green
}

function Invoke-Build {
    Write-Host "Building skill package: $SkillName" -ForegroundColor Yellow

    # Create directories
    $skillDir = Join-Path $BuildDir $SkillName
    New-Item -ItemType Directory -Force -Path $skillDir | Out-Null
    New-Item -ItemType Directory -Force -Path "$skillDir/references" | Out-Null
    New-Item -ItemType Directory -Force -Path "$skillDir/scripts" | Out-Null
    New-Item -ItemType Directory -Force -Path "$skillDir/config" | Out-Null

    # Copy main skill file
    Copy-Item "src/SKILL.md" $skillDir

    # Copy reference files
    Copy-Item "src/references/*.md" "$skillDir/references/"

    # Copy scripts
    Copy-Item "src/scripts/*.py" "$skillDir/scripts/"

    # Copy config
    Copy-Item "src/config/domains.json" "$skillDir/config/"

    # Copy documentation
    @("README.md", "LICENSE", "CHANGELOG.md") | ForEach-Object {
        if (Test-Path $_) {
            Copy-Item $_ $skillDir
        }
    }

    Write-Host "Build complete: $skillDir" -ForegroundColor Green
}

function Invoke-BuildCombined {
    Write-Host "Building combined single-file skill..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    $outputFile = Join-Path $BuildDir "$SkillName-combined.md"

    # Start with SKILL.md
    $content = Get-Content "src/SKILL.md" -Raw
    $content += "`n`n---`n`n# Bundled References`n"

    # Append each reference file
    Get-ChildItem "src/references/*.md" | ForEach-Object {
        $content += "`n---`n`n"
        $content += Get-Content $_.FullName -Raw
    }

    Set-Content -Path $outputFile -Value $content

    Write-Host "Combined skill created: $outputFile" -ForegroundColor Green
}

function Invoke-Package {
    Invoke-Build

    Write-Host "Packaging skill as zip..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

    $zipFile = Join-Path (Resolve-Path $DistDir) "$SkillName-v$Version.zip"
    $sourcePath = Resolve-Path (Join-Path $BuildDir $SkillName)

    # Remove existing zip if present
    if (Test-Path $zipFile) {
        Remove-Item $zipFile
    }

    # Use .NET ZipFile with forward slashes for cross-platform compatibility
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($zipFile, 'Create')

    try {
        Get-ChildItem -Path $sourcePath -Recurse -File | ForEach-Object {
            $relativePath = $_.FullName.Substring($sourcePath.Path.Length + 1)
            # Convert backslashes to forward slashes for cross-platform compatibility
            $entryName = "$SkillName/" + ($relativePath -replace '\\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $entryName) | Out-Null
        }
    }
    finally {
        $zip.Dispose()
    }

    Write-Host "Package created: $zipFile" -ForegroundColor Green
}

function Invoke-PackageCombined {
    Invoke-BuildCombined

    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

    $source = Join-Path $BuildDir "$SkillName-combined.md"
    $dest = Join-Path $DistDir "$SkillName-combined.md"

    Copy-Item $source $dest

    Write-Host "Combined skill copied to: $dest" -ForegroundColor Green
}

function Invoke-Validate {
    Write-Host "Validating skill structure..." -ForegroundColor Yellow

    $errors = @()

    # Check SKILL.md exists
    if (-not (Test-Path "src/SKILL.md")) {
        $errors += "ERROR: src/SKILL.md not found"
    } else {
        $content = Get-Content "src/SKILL.md" -Raw
        if ($content -notmatch "(?m)^name:") {
            $errors += "ERROR: SKILL.md missing 'name' in frontmatter"
        }
        if ($content -notmatch "(?m)^description:") {
            $errors += "ERROR: SKILL.md missing 'description' in frontmatter"
        }
    }

    # Check directories
    if (-not (Test-Path "src/references")) {
        $errors += "ERROR: src/references/ directory not found"
    }
    if (-not (Test-Path "src/scripts")) {
        $errors += "ERROR: src/scripts/ directory not found"
    }

    if ($errors.Count -gt 0) {
        $errors | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        exit 1
    }

    Write-Host "Validation passed!" -ForegroundColor Green
}

function Invoke-Lint {
    Write-Host "Checking Python syntax..." -ForegroundColor Yellow
    $failed = $false
    Get-ChildItem src/scripts/*.py | ForEach-Object {
        Write-Host "  py_compile $($_.Name)"
        python -m py_compile $_.FullName
        if ($LASTEXITCODE -ne 0) { $failed = $true }
    }
    if ($failed) {
        Write-Host "Syntax check failed!" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Syntax check passed!" -ForegroundColor Green
    }
}

function Invoke-Clean {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow

    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
    if (Test-Path $DistDir) {
        Remove-Item -Recurse -Force $DistDir
    }
    if (Test-Path "hypotheses.json") {
        Remove-Item "hypotheses.json"
    }

    Write-Host "Clean complete" -ForegroundColor Green
}

function Invoke-SyncSkill {
    $skillDest = Join-Path $env:USERPROFILE ".claude" "skills" $SkillName
    Write-Host "Syncing skill to $skillDest..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path "$skillDest/references" | Out-Null
    New-Item -ItemType Directory -Force -Path "$skillDest/scripts" | Out-Null
    New-Item -ItemType Directory -Force -Path "$skillDest/config" | Out-Null

    Copy-Item "src/SKILL.md" $skillDest
    Copy-Item "src/references/*.md" "$skillDest/references/"
    Copy-Item "src/scripts/*.py" "$skillDest/scripts/"
    Copy-Item "src/config/domains.json" "$skillDest/config/"

    Write-Host "Skill synced to $skillDest" -ForegroundColor Green
}

function Invoke-All {
    Invoke-Package
}

function Invoke-Install {
    Invoke-Build
    Write-Host "Installing skill..." -ForegroundColor Yellow
    Write-Host "Copy $BuildDir/$SkillName to your Claude skills directory"
    Write-Host "Or use: .\build.ps1 package then extract dist/$SkillName-v$Version.zip to ~/.claude/skills/"
}

function Invoke-PackageTar {
    Invoke-Build

    Write-Host "Packaging skill as tarball..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

    $tarFile = "$DistDir/$SkillName-v$Version.tar.gz"
    tar -czvf $tarFile -C $BuildDir $SkillName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tarball creation failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "Package created: $tarFile" -ForegroundColor Green
}

function Invoke-Test {
    Invoke-Lint

    # Unit tests
    Write-Host "Running unit tests..." -ForegroundColor Yellow
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Unit tests failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Unit tests passed!" -ForegroundColor Green

    # Smoke tests
    Write-Host "Running --help smoke tests..." -ForegroundColor Yellow
    $failed = $false
    Get-ChildItem src/scripts/*.py | ForEach-Object {
        Write-Host "  $($_.Name) --help"
        python $_.FullName --help > $null
        if ($LASTEXITCODE -ne 0) { $failed = $true }
    }
    if ($failed) {
        Write-Host "Smoke tests failed!" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Smoke tests passed!" -ForegroundColor Green
    }
}

function Invoke-List {
    Invoke-Build

    Write-Host "Package contents:" -ForegroundColor Cyan
    Get-ChildItem -Recurse (Join-Path $BuildDir $SkillName) |
        Where-Object { -not $_.PSIsContainer } |
        ForEach-Object { $_.FullName.Replace((Get-Location).Path + "\", "") }
}

# Execute command
switch ($Command.ToLower()) {
    "all"             { Invoke-All }
    "build"           { Invoke-Build }
    "build-combined"  { Invoke-BuildCombined }
    "package"         { Invoke-Package }
    "package-combined" { Invoke-PackageCombined }
    "package-tar"     { Invoke-PackageTar }
    "validate"        { Invoke-Validate }
    "lint"            { Invoke-Lint }
    "test"            { Invoke-Test }
    "clean"           { Invoke-Clean }
    "list"            { Invoke-List }
    "install"         { Invoke-Install }
    "sync-skill"      { Invoke-SyncSkill }
    "help"            { Show-Help }
    default           { Show-Help }
}
