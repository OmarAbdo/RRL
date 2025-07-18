# This is a PowerShell script.
# To run it, you may first need to run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Then execute the script: .\Task_4\build_and_push.ps1

# Get the directory where the script is located and change to it
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Adding Google Cloud SDK to PATH for this session..."
$env:Path = "C:\Users\Omar\AppData\Local\Google\Cloud SDK\bin;" + $env:Path

Write-Host "Setting the image URI..."
$env:IMAGE_URI="us-central1-docker.pkg.dev/gen-lang-client-0839523272/r-rl-repo/r-rl-vertex-job:latest"

Write-Host "Authenticating Docker with Google Cloud..."
# Use the call operator (&) to execute the command with its full path
& "C:\Users\Omar\AppData\Local\Google\Cloud SDK\gcloud.cmd" auth configure-docker us-central1-docker.pkg.dev --quiet

Write-Host "Building the Docker image..."
docker build -t $env:IMAGE_URI .

Write-Host "Pushing the Docker image to Artifact Registry..."
docker push $env:IMAGE_URI

Write-Host "Script finished successfully!"
