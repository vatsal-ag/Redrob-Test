# TrueBoard (Team Vatsal and Krishna) 🚀

TrueBoard is a 7-Layer Behavior-Aware Contextual Ranking Engine built for the India Runs Data & AI Challenge.

## Links
- **Interactive Sandbox:** [Update with HuggingFace Space Link]
- **Landing Page:** [https://vatsal-ag.github.io/Redrob-Test/](https://vatsal-ag.github.io/Redrob-Test/)

## 🚀 How to Run the Full 100k Dataset Locally

Because GitHub blocks files over 100MB, the 487MB `candidates.jsonl` database and our 153MB pre-computed `embeddings_cache.npy` are not included in this repo.

To run the full 100,000 candidate ranking on your machine:
1. Place the official `[PUB] India_runs_data_and_ai_challenge` folder into the root of this repository.
2. Run `python app.py` (to start the web server) OR `python run_pipeline.py` (to generate the final CSV).
*(Our engine is smart enough to detect the missing cache and will automatically generate it in the background if the dataset is present!)*

## 📦 How to generate `team_submission.csv` (For Judges)

1. Ensure the `[PUB] India_runs_data_and_ai_challenge` folder is in the root directory.
2. Ensure you have installed the requirements: `pip install -r requirements.txt`
3. Run the offline pipeline:
```bash
python run_pipeline.py
```
This will automatically generate a perfectly valid, monotone-checked `team_submission.csv` in the root directory.

## ☁️ Cloud Sandbox Deployment
We have provided a `Dockerfile` and `requirements.txt` perfectly configured for a **HuggingFace Docker Space**. 
To deploy your own instance:
1. Create a new Docker Space on HuggingFace.
2. Upload the files in this repository.
3. The Space will automatically build and host the interactive web UI on port 7860!