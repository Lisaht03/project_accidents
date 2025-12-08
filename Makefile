# ============================================
# ### DOCKER LOCAL
# ============================================

# Build the container locally for development
# Usage: Builds the image using the current directory (.)
docker build --tag=$IMAGE:dev .

# Run the container locally
# Usage: Runs interactively (-it), sets PORT env var, and maps port 8000
docker run -it -e PORT=8000 -p 8000:8000 $IMAGE:dev


# ============================================
# ## DOCKER DEPLOYMENT GCP
# ============================================

# --- Step 1: Authentication (EXECUTE ONLY ONCE) ---
# Configure Docker to use gcloud credentials for the specific region
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev

# --- Step 2: Create Repository (EXECUTE ONLY ONCE) ---
# Create the Artifact Registry repo to store Docker images
gcloud artifacts repositories create $ARTIFACTSREPO --repository-format=docker \
  --location=$GCP_REGION --description="Repository for storing images"

# --- Step 3: Build for Production ---

# OPTION A: For Windows or Mac with Intel Chips
# Builds and tags the image directly for the remote registry
docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$IMAGE:prod .

# OPTION B: ⚠️ SPECIFICALLY FOR MAC M1 / M2 / M3 (Apple Silicon)
# We must force '--platform linux/amd64' because Cloud Run requires AMD64 architecture.
# If you don't use this flag on a Mac M-chip, the deploy will fail with "Exec format error".
docker build --platform linux/amd64 -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$IMAGE:prod .


# --- Step 4: Push Image ---
# Uploads the production image to Google Artifact Registry
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$IMAGE:prod

# --- Step 5: Deploy ---
# Deploys the image to Google Cloud Run
gcloud run deploy $INSTANCE --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$IMAGE:prod \
  --memory $MEMORY --region $GCP_REGION --allow-unauthenticated


# ============================================
# ## CLEANUP & MAINTENANCE
# ============================================

# Disable the Service (Scale to Zero)
# Adjusts configuration to scale down to zero instances.
# No resources are used, avoiding costs for idle time.
gcloud run services update $INSTANCE --min-instances=0 --region $GCP_REGION

# Delete the Service
# Permanently removes the service from Cloud Run.
gcloud run services delete $INSTANCE --region $GCP_REGION
