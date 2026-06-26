# TrueBoard AI — Redrob Hackathon Submission
### Vatsal Agarwal | B.Tech 2nd Year

## What this is
A 7-layer behavior-aware candidate ranking engine for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.

Ranks candidates from a 100,000-record pool against a "Senior AI Engineer — Founding Team" job description, avoiding the keyword-stuffing trap and surfacing hidden gems using behavioral signals.

**Live Demo**: [TrueBoard Dashboard](https://vatsal-ag.github.io/Redrob-Test/)

## Features & Out-of-the-Box Detectors
We go way beyond semantic similarity by analyzing human behavior and career trajectories:
- **Title Inflation Detector**: Flags candidates with senior titles but suspiciously low years of experience.
- **Peter Principle Detector**: Identifies ICs promoted to management who had very short tenures.
- **Over-Employment (OE) Siren**: Detects overlapping "Present" roles to warn of moonlighting.
- **Wartime Survivor**: Flags candidates with long-term tenure at a single company through tough periods.
- **AI Clone Chatbot**: Simulates a conversation with the candidate, complete with dynamic salary expectations based on their specific title and YOE!

## Files
| File | Description |
|---|---|
| `app.py` | Flask web app with live ranking dashboard and behavioral logic |
| `run_pipeline.py` | **Offline batch pipeline** — run this to score all 100k candidates and generate `team_submission.csv` |
| `templates/Index.html` | Premium dark UI dashboard |
| `sample_candidates.json` | 50-candidate sample for quick local testing |
| `test_dataset.json` | 10 hand-crafted edge-case candidates for engine validation |
| `team_submission.csv` | Final submission (top 100 candidates, validator-compliant) |
| `mock_data.json` | Captured results used for the GitHub Pages static site demo |

## Scoring Architecture (7 Layers)
1. **Semantic Match** — `all-MiniLM-L6-v2` cosine similarity on headline + summary + all career descriptions + skills.
2. **Profile Gate** — Title trap kill-switch (Marketing/HR/Civil → ×0.06), career recovery, YOE sweet-spot (5–9 yrs), services-firm penalty.
3. **Skill Authenticity** — Cross-references each claimed skill against career history descriptions (kills keyword stuffers).
4. **Behavioral Score** — 23 `redrob_signals`: response rate, last active date, GitHub activity, open-to-work, interview completion, offer acceptance, notice period, etc.
5. **Education Tier Bonus** — IIT/IISc tier_1 → ×1.18.
6. **Assessment Bonus** — Platform-proctored skill test scores (cannot be faked).
7. **Min-Max Normalisation** → scores in [0.0001, 1.0000], tie-break by `candidate_id ASC`.

## How to run

### Web Dashboard (live ranking)
```bash
pip install -r requirements.txt
python3 app.py
# Open http://localhost:5001 — paste JD — click Analyze
```

### Offline pipeline (generates team_submission.csv from 100k candidates)
```bash
# Put candidates.jsonl in the root or data folder
python3 run_pipeline.py
```