#!/usr/bin/env python3
"""
TrueBoard AI — Offline Ranking Pipeline (Dynamic Universal Engine)
================================================================
Reads the real candidates.jsonl (100k records), scores all candidates using
the 7-layer dynamic engine, and writes a validator-compliant team_submission.csv.

Usage:
    python3 run_pipeline.py

Output:
    team_submission.csv  (top 100, validator-compliant)
"""

import os, json, csv, time, re
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Paths
CHALLENGE_DIR = os.path.join(os.path.dirname(__file__),
    '[PUB] India_runs_data_and_ai_challenge',
    'India_runs_data_and_ai_challenge')

JSONL_PATH   = os.path.join(CHALLENGE_DIR, 'candidates.jsonl')
OUTPUT_CSV   = os.path.join(os.path.dirname(__file__), 'team_submission.csv')
BATCH_SIZE   = 512

# Target Job Description (Can be changed to test any role)
JD_TEXT = """
Senior AI Engineer — Founding Team — Redrob

We are building the intelligence layer of Redrob's product: the ranking, retrieval,
and matching systems that decide what recruiters see when they search for candidates
and what candidates see when they search for roles.

Absolutely required:
- Production experience with embeddings-based retrieval systems (sentence-transformers,
  OpenAI embeddings, BGE, E5 or similar) deployed to real users. Must have handled
  embedding drift, index refresh, retrieval-quality regression in production.
- Production experience with vector databases or hybrid search: Pinecone, Weaviate,
  Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
- Strong Python, code quality matters.
- Hands-on experience designing evaluation frameworks for ranking systems: NDCG, MRR,
  MAP, offline-to-online correlation, A/B test interpretation.
- 5-9 years total experience, tilted toward product companies not services firms.

Nice to have:
- LLM fine-tuning (LoRA, QLoRA, PEFT, RLHF).
- Learning-to-rank models (XGBoost-based or neural).
- NLP, recommendation systems, information retrieval.
- Open-source contributions in AI/ML.
- Prior HR-tech, recruiting-tech, marketplace product experience.
"""

# Helpers
def days_since(date_str):
    if not date_str: return 9999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except ValueError:
        return 9999

def build_text(c):
    p = c.get('profile', {})
    parts = [p.get('headline','')]
    for j in c.get('career_history', []):
        parts.append(j.get('title',''))
    parts.append(', '.join(s['name'] for s in c.get('skills',[])))
    return ' '.join(x for x in parts if x)

# Dynamic JD Parsers
def extract_jd_features(jd_text):
    text = jd_text.lower()
    
    stopwords = {'of', 'to', 'in', 'on', 'at', 'as', 'by', 'is', 'it', 'or', 'be', 'an', 'we', 'us', 'if', 'do', 'am', 'this', 'that', 'with', 'from', 'your', 'have', 'will', 'more', 'about', 'which', 'what', 'when', 'where', 'they', 'their', 'there', 'and', 'for', 'the', 'years', 'year', 'experience', 'exp', 'plus'}

    # 1. Extract JD Title Words (assume first line contains the title)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    jd_title_words = set()
    if lines:
        title_line = re.sub(r'[^\w\s]', ' ', lines[0])
        jd_title_words = {w for w in title_line.split() if len(w) >= 2 and w not in stopwords and not w.isdigit()}

        # Synonym expansion to handle acronyms vs full words
        synonyms = {
            'human': ['hr'], 'resources': ['hr'], 'hr': ['human', 'resources'],
            'vice': ['vp'], 'president': ['vp'], 'vp': ['vice', 'president'],
            'ceo': ['chief', 'executive', 'officer'],
            'cto': ['chief', 'technology', 'officer'],
            'chief': ['ceo', 'cto', 'cfo', 'coo', 'cmo'],
            'ux': ['user', 'experience'], 'ui': ['user', 'interface'],
            'qa': ['quality', 'assurance'], 'qc': ['quality', 'control'],
            'ml': ['machine', 'learning'], 'ai': ['artificial', 'intelligence'],
            'swe': ['software', 'engineer'], 'sde': ['software', 'developer'],
            'devops': ['development', 'operations'],
            'frontend': ['front', 'end'], 'backend': ['back', 'end'],
            'rep': ['representative'], 'sales': ['sdr', 'bdr', 'ae']
        }
        expanded = set(jd_title_words)
        for w in jd_title_words:
            if w in synonyms:
                expanded.update(synonyms[w])
        jd_title_words = expanded

    # 2. Extract YoE
    yoe_match = re.search(r'(\d+)\s*(?:-|to|and)\s*(\d+)\s*years', text)
    if yoe_match:
        min_yoe = float(yoe_match.group(1))
        max_yoe = float(yoe_match.group(2))
    else:
        yoe_match_single = re.search(r'(\d+)\+?\s*years', text)
        if yoe_match_single:
            min_yoe = float(yoe_match_single.group(1))
            max_yoe = min_yoe + 5.0
        else:
            min_yoe, max_yoe = None, None

    # 3. Extract keywords (all alphanumeric words > 3 chars)
    words = re.findall(r'\b[a-z]{3,}\b', text)
    keywords = set(w for w in words if w not in stopwords)
    
    return {
        'jd_title_words': jd_title_words,
        'min_yoe': min_yoe,
        'max_yoe': max_yoe,
        'keywords': keywords,
    }

# Scoring Layers (Imported/Duplicated from app.py)
def dynamic_profile_gate(c, jd_features):
    p = c.get('profile', {})
    yoe = p.get('years_of_experience', 0.0)
    title = p.get('current_title', '').lower()
    headline = p.get('headline', '').lower()
    m = 1.0

    # 1. Explicit Title / Headline Relevance Boost
    jd_title_words = jd_features['jd_title_words']
    if jd_title_words:
        title_headline = f"{title} {headline}"
        title_headline_words = set(re.findall(r'\b[a-z]{2,}\b', title_headline))
        overlap = sum(1 for w in jd_title_words if w in title_headline_words)
        
        if overlap == 0:
            m *= 0.2  # Massive penalty if their current title/headline has literally nothing to do with the JD title!
        elif overlap == len(jd_title_words):
            m *= 4.0  # Massive boost for perfect title match
        else:
            m *= 1.5  # Soft boost for partial title match

    # 2. YoE Gate
    min_y = jd_features['min_yoe']
    max_y = jd_features['max_yoe']
    
    if min_y is not None and max_y is not None:
        if yoe < min_y - 2.0:
            m *= 0.45
        elif yoe < min_y:
            m *= 0.80
        elif yoe <= max_y + 2.0:
            m *= 1.00 
        else:
            m *= 0.85

    return m

def dynamic_skill_authenticity(c, jd_features, evidence_text):
    skills = c.get('skills', [])
    if not skills: return 0.65
    
    keywords = jd_features['keywords']
    relevant_skills = [s['name'].lower() for s in skills if any(w in s['name'].lower() for w in keywords)]
    
    if not relevant_skills: return 0.80
        
    verified = sum(1 for s in relevant_skills if s in evidence_text or s.replace(' ', '') in evidence_text)
    return verified / len(relevant_skills)

def dynamic_assessment_bonus(c, jd_features):
    scores = c.get('redrob_signals', {}).get('skill_assessment_scores', {})
    if not scores: return 1.00
    
    keywords = jd_features['keywords']
    rel = [v for k,v in scores.items() if any(w in k.lower() for w in keywords)]
    
    all_s = list(scores.values())
    weighted = rel*2 + [s for s in all_s if s not in rel]
    avg = sum(weighted)/len(weighted) if weighted else 50.0
    best = max(all_s)
    
    if avg > 72:     m = 1.28
    elif avg > 58:   m = 1.16
    elif avg > 42:   m = 1.06
    elif avg < 22:   m = 0.82
    else:            m = 1.00
    if best > 88:    m = min(m*1.06, 1.45)
    return m

def behavioral_score(c):
    sig = c.get('redrob_signals', {})
    if not sig: return 1.0
    m = 1.0
    
    rr  = sig.get('recruiter_response_rate', 0.5)
    if rr < 0.15:    m *= 0.20
    elif rr < 0.25:  m *= 0.35
    elif rr < 0.50:  m *= 0.75
    elif rr > 0.85:  m *= 1.22
    elif rr > 0.70:  m *= 1.10

    icr = sig.get('interview_completion_rate', 0.5)
    if icr < 0.35:   m *= 0.50
    elif icr < 0.55: m *= 0.82
    elif icr > 0.85: m *= 1.12

    gh = sig.get('github_activity_score', -1)
    if gh == -1:     m *= 0.88
    elif gh > 60:    m *= 1.22
    elif gh > 40:    m *= 1.15
    elif gh > 20:    m *= 1.07

    days = days_since(sig.get('last_active_date', ''))
    if days > 180:   m *= 0.25
    elif days > 120: m *= 0.45
    elif days > 90:  m *= 0.68
    elif days > 45:  m *= 0.87
    elif days <= 7:  m *= 1.12
    elif days <= 14: m *= 1.06

    if sig.get('open_to_work_flag', False): m *= 1.14

    oar = sig.get('offer_acceptance_rate', 0.5)
    if oar < 0.25:   m *= 0.72
    elif oar > 0.80: m *= 1.06

    art = sig.get('avg_response_time_hours', 48)
    if art > 120:    m *= 0.84
    elif art < 12:   m *= 1.07

    pcs = sig.get('profile_completeness_score', 70)
    if pcs > 88:     m *= 1.06
    elif pcs < 45:   m *= 0.88

    if sig.get('verified_email') and sig.get('verified_phone'): m *= 1.05

    notice = sig.get('notice_period_days', 60)
    if notice <= 15:   m *= 1.08
    elif notice <= 30: m *= 1.03
    elif notice > 90:  m *= 0.88

    if sig.get('linkedin_connected', False): m *= 1.03
    saved = sig.get('saved_by_recruiters_30d', 0)
    if saved > 10:   m *= 1.05
    elif saved > 5:  m *= 1.02
    
    return m

def education_bonus(c):
    best = 4
    for edu in c.get('education', []):
        try: best = min(best, int(edu.get('tier','tier_4').replace('tier_','')))
        except: pass
    return {1:1.18, 2:1.08, 3:1.00, 4:1.00}.get(best, 1.00)

def make_reasoning(c, semantic, auth):
    p = c.get('profile', {})
    sig = c.get('redrob_signals', {})
    rr = sig.get('recruiter_response_rate', 0)
    days = days_since(sig.get('last_active_date', ''))
    otw = 'Yes' if sig.get('open_to_work_flag') else 'No'
    return (
        f"{p.get('current_title','N/A')}, {p.get('years_of_experience',0)}yrs. "
        f"Semantic: {semantic:.1f}/100. "
        f"Skill auth: {int(auth*100)}%. "
        f"Response rate: {int(rr*100)}%. "
        f"Last active: {days}d ago. "
        f"Open to work: {otw}."
    )

# Offline Dataset Loading
def run():
    t0 = time.time()

    import torch
    print("Loading AI model...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    print("Parsing JD...")
    jd_features = extract_jd_features(JD_TEXT)
    jd_norm = model.encode([JD_TEXT], normalize_embeddings=True)

    print(f"Reading {JSONL_PATH}...")
    candidates = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    total = len(candidates)
    print(f"Loaded {total:,} candidates. Building text representations...")

    EMBED_CACHE = os.path.join(os.path.dirname(__file__), 'embeddings_cache.npy')
    if os.path.exists(EMBED_CACHE):
        print("Loading pre-computed embeddings from cache...")
        all_embeddings = np.load(EMBED_CACHE)
    else:
        texts = [build_text(c) for c in candidates]
        print(f"WARNING: Cache not found. Encoding candidate texts in batches of {BATCH_SIZE} (this will take 2-3 mins)...")
        all_embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print("Saving embeddings cache to disk...")
        np.save(EMBED_CACHE, all_embeddings)
        print("Saved cache to embeddings_cache.npy!")

    print("Scoring all candidates dynamically against JD...")
    results = []
    for i, c in enumerate(candidates):
        evidence = ' '.join(
            f"{j.get('title','')} {j.get('description','')}".lower()
            for j in c.get('career_history', [])
        ) + ' ' + c.get('profile', {}).get('summary', '').lower()

        sem = float(np.dot(jd_norm[0], all_embeddings[i])) * 100
        
        pm   = dynamic_profile_gate(c, jd_features)
        auth = dynamic_skill_authenticity(c, jd_features, evidence)
        am   = dynamic_assessment_bonus(c, jd_features)
        bm   = behavioral_score(c)
        em   = education_bonus(c)

        raw = sem * pm * auth * bm * em * am

        results.append({
            'candidate_id': c['candidate_id'],
            'raw_score': raw,
            'semantic': sem,
            'auth': auth,
            'reasoning': make_reasoning(c, sem, auth),
        })

    print("Normalising scores and ranking...")
    raw_scores = [r['raw_score'] for r in results]
    s_min, s_max = min(raw_scores), max(raw_scores)
    span = s_max - s_min if s_max != s_min else 1.0

    for r in results:
        norm = (r['raw_score'] - s_min) / span
        r['score'] = round(max(0.0001, min(1.0000, norm)), 4)

    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    top100 = results[:100]

    for i, r in enumerate(top100):
        r['rank'] = i + 1

    print(f"\n{'='*60}")
    print("TOP 10 CANDIDATES:")
    print(f"{'='*60}")
    for r in top100[:10]:
        print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']:.4f}  sem={r['semantic']:.1f}  {r['reasoning'][:60]}")

    print(f"\nWriting {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in top100:
            w.writerow([r['candidate_id'], r['rank'], f"{r['score']:.4f}", r['reasoning']])

    scores = [r['score'] for r in top100]
    violations = sum(1 for i in range(len(scores)-1) if scores[i] < scores[i+1])

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY:")
    print(f"  Rows written : {len(top100)}")
    print(f"  Score range  : {min(scores):.4f} – {max(scores):.4f}")
    print(f"  Monotone     : {'✅ PASS' if violations == 0 else f'❌ FAIL ({violations} violations)'}")
    print(f"  Time elapsed : {time.time()-t0:.1f}s")
    print(f"{'='*60}")
    print(f"\n✅  team_submission.csv is ready.")

if __name__ == '__main__':
    run()
