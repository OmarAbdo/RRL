$PROJECT_ID="gen-lang-client-0839523272"
$REGION="europe-west4"
$REPO_NAME="r-rl-repo-ew4"
$IMAGE_NAME="r-rl-vertex-job-d-task4"
$IMAGE_TAG="latest"

$IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

Write-Host "=================================================="
Write-Host "Building and Pushing Docker Image for Vertex AI"
Write-Host "=================================================="
Write-Host "Project: $PROJECT_ID"
Write-Host "Region: $REGION"
Write-Host "Image URI: $IMAGE_URI"
Write-Host "=================================================="

# Authenticate Docker to the Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Build the Docker image
docker build -t $IMAGE_URI .

# Push the Docker image to the Artifact Registry
docker push $IMAGE_URI

Write-Host ""
Write-Host "=================================================="
Write-Host "Image pushed successfully!"
Write-Host "You can now submit the Vertex AI job using:"
Write-Host "python launch_vertex_job.py"
Write-Host "=================================================="
