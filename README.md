# CardioSight Clinical Risk Console

A GitHub-ready Flask project for Render deployment. This app presents a polished hospital-style UI for cardiovascular disease risk screening using a saved machine learning pipeline.

## Deployment-ready files
- `app.py`
- `requirements.txt`
- `.python-version`
- `render.yaml`
- `templates/index.html`
- `static/styles.css`
- `model/` (put the saved `.joblib` model here)
- `model/model_metadata.json`

## Local run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Render deployment
1. Create a new GitHub repository.
2. Upload the files in this folder.
3. Commit and push.
4. In Render, create a new Web Service from the GitHub repo.
5. Render will read `render.yaml`, or you can set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`

## Important
The app works best when you place your trained model file here:
`model/cvd_logreg_pipeline.joblib`

And the metadata file here:
`model/model_metadata.json`

If those files are missing, the UI will still open with demo fallback behavior, but real deployment should always include the saved model.
