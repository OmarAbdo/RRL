@echo OFF
set "PROJECT_ID=gen-lang-client-0839523272"
set "REGION=europe-west4"
set "REPO_NAME=r-rl-repo-ew4"
set "IMAGE_NAME=r-rl-vertex-job-d-task4"
set "IMAGE_TAG=latest"

set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO_NAME%/%IMAGE_NAME%:%IMAGE_TAG%"

echo ==================================================
echo Building and Pushing Docker Image for Vertex AI
echo ==================================================
echo Project: %PROJECT_ID%
echo Region: %REGION%
echo Image URI: %IMAGE_URI%
echo ==================================================

REM Authenticate Docker to the Artifact Registry
gcloud auth configure-docker %REGION%-docker.pkg.dev

REM Build the Docker image
docker build -t %IMAGE_URI% .

REM Push the Docker image to the Artifact Registry
docker push %IMAGE_URI%

echo.
echo ==================================================
echo Image pushed successfully!
echo You can now submit the Vertex AI job using:
echo python launch_vertex_job.py
echo ==================================================
