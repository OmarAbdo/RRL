@echo OFF
echo Adding Google Cloud SDK to PATH...
set "PATH=C:\Users\Omar\AppData\Local\Google\Cloud SDK\bin;%PATH%"

echo Setting the image URI...
set IMAGE_URI=us-central1-docker.pkg.dev/gen-lang-client-0839523272/r-rl-repo/r-rl-vertex-job:latest

echo Changing to the Task_4 directory...
cd Task_4

echo Building the Docker image...
docker build -t %IMAGE_URI% .

echo Authenticating Docker with Google Cloud...
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

echo Pushing the Docker image to Artifact Registry...
docker push %IMAGE_URI%

echo.
echo Script finished.
