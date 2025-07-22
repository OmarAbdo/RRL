import google.cloud.aiplatform as aip

# --- CONFIGURATION ---
PROJECT_ID = "gen-lang-client-0839523272"
REGION = "europe-west4"
STAGING_BUCKET = "gs://r-rl-staging-bucket-ew4-gen-lang-client-0839523272"
IMAGE_URI = "europe-west4-docker.pkg.dev/gen-lang-client-0839523272/r-rl-repo-ew4/r-rl-vertex-job-d-task4:latest"
JOB_NAME = "r-rl-training-job-d-task4"

# --- JOB SUBMISSION ---
def main():
    # Initialize the Vertex AI SDK
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    # Define the job
    job = aip.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-32",
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": IMAGE_URI,
                    "command": [],
                    "args": [],
                },
            }
        ],
    )

    print(f"Submitting Vertex AI training job '{JOB_NAME}'...")
    # Submit the job for execution
    job.run()

    print("Job submitted successfully. You can monitor its progress in the Google Cloud Console.")
    print(f"Job resource name: {job.resource_name}")

if __name__ == "__main__":
    main()
