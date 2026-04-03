# Render-ready CVD App

## Use this exact version on Render

1. Create a new GitHub repository
2. Upload all files from this folder
3. Commit and push
4. In Render:
   - New + -> Web Service
   - Connect the GitHub repo
   - Choose Docker environment
5. Deploy

Do not enter manual build or start commands.
Render will use the Dockerfile automatically.

If you have a real saved model, place it here:
model/cvd_logreg_pipeline.joblib

If you do not add the model file, the app still deploys and runs using safe fallback scoring so you can present the UI immediately.
