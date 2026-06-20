$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $Root
try {
  python scripts\render_submission_assets.py
  Push-Location paper
  try {
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed with exit code $LASTEXITCODE" }
    bibtex main
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed with exit code $LASTEXITCODE" }
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 2 failed with exit code $LASTEXITCODE" }
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 3 failed with exit code $LASTEXITCODE" }
  }
  finally {
    Pop-Location
  }
  $dest = "C:\Users\wangz\Downloads\71.pdf"
  Copy-Item -LiteralPath "paper\main.pdf" -Destination $dest -Force
  if (Test-Path -LiteralPath "C:\Users\wangz\Desktop\71.pdf") {
    throw "Desktop hygiene violation: C:\Users\wangz\Desktop\71.pdf exists"
  }
  Write-Host "Built $dest"
}
finally {
  Pop-Location
}
