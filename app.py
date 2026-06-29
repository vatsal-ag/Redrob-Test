import os
import json
import csv
import time
import re
import numpy as np
from datetime import date, datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(__file__)
JSONL_PATH   = os.path.join(BASE_DIR,
    '[PUB] India_runs_data_and_ai_challenge',
    'India_runs_data_and_ai_challenge',
    'candidates.jsonl')
SAMPLE_PATH  = os.path.join(BASE_DIR, 'sample_candidates.json')
OUTPUT_CSV   = os.path.join(BASE_DIR, 'team_submission.csv')
EMBED_CACHE  = os.path.join(BASE_DIR, 'embeddings_cache.npy')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Dynamic JD Parsers
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Scoring Layers
# ---------------------------------------------------------------------------
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
        # Tokenize to prevent substring bugs (e.g. 'hr' matching 'three')
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
            # Bonus for exceeding the minimum requirement (up to 10 years extra = 1.5x multiplier)
            extra_years = min(max(0, yoe - min_y), 10.0)
            m *= (1.0 + (extra_years * 0.05))
        else:
            m *= 0.85 # over-senior

    return m

def dynamic_skill_authenticity(c, jd_features, evidence_text):
    skills = c.get('skills', [])
    if not skills: return 0.65
    
    keywords = jd_features['keywords']
    # Filter candidate skills to only those that overlap with JD keywords
    relevant_skills = [s['name'].lower() for s in skills if any(w in s['name'].lower() for w in keywords)]
    
    if not relevant_skills:
        return 0.80  # neutral if no exact skill intersection with JD
        
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

# ---------------------------------------------------------------------------
# STARTUP: Load candidates + pre-compute embeddings (runs ONCE)
# ---------------------------------------------------------------------------
    print("============================================================")
    print("Engine — Initializing (Dynamic Universal Engine)")
    print("============================================================")
import torch
t0 = time.time()
print("Loading sentence-transformer model...")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
ai_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("Loading candidates...")
CANDIDATES = []
if os.path.exists(JSONL_PATH):
    print("Found full 100k dataset. Loading from candidates.jsonl...")
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: CANDIDATES.append(json.loads(line))
    print(f"  Loaded {len(CANDIDATES):,} candidates.")
    
    if os.path.exists(EMBED_CACHE):
        print("Loading pre-computed embeddings from cache...")
        CANDIDATE_EMBEDDINGS = np.load(EMBED_CACHE)
        print(f"  Loaded embeddings shape: {CANDIDATE_EMBEDDINGS.shape}")
    else:
        print("WARNING: Embedding cache not found. Building embeddings on the fly (this will take 2-3 minutes)...")
        texts = [build_text(c) for c in CANDIDATES]
        CANDIDATE_EMBEDDINGS = ai_model.encode(
            texts, batch_size=512, show_progress_bar=True, normalize_embeddings=True
        )
        print("Saving embeddings cache to disk...")
        np.save(EMBED_CACHE, CANDIDATE_EMBEDDINGS)
        print("Saved cache to embeddings_cache.npy!")
elif os.path.exists(SAMPLE_PATH):
    print("WARNING: 100k dataset not found. Falling back to sample dataset.")
    with open(SAMPLE_PATH, 'r', encoding='utf-8') as f:
        CANDIDATES = json.load(f)
    print(f"  Loaded {len(CANDIDATES)} candidates from sample_candidates.json")
    texts = [build_text(c) for c in CANDIDATES]
    CANDIDATE_EMBEDDINGS = ai_model.encode(
        texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True
    )
else:
    print("  WARNING: No candidate data found!")
    CANDIDATE_EMBEDDINGS = np.array([])

# Pre-compute JD-agnostic scores & evidence text for ultra-fast query times
print("Pre-computing behavioral scores and evidence text...")
CANDIDATE_CACHE = []
for c in CANDIDATES:
    evidence = ' '.join(
        f"{j.get('title','')} {j.get('description','')}".lower()
        for j in c.get('career_history', [])
    ) + ' ' + c.get('profile', {}).get('summary', '').lower()
    
    CANDIDATE_CACHE.append({
        'evidence': evidence,
        'bm': behavioral_score(c),
        'em': education_bonus(c),
    })

print(f"✅ Startup complete in {time.time()-t0:.1f}s — server is ready!\n")

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/rank', methods=['POST'])
def rank_candidates():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON.'}), 400

    job_desc = request.json.get('jd_text', '').strip()
    company_name = request.json.get('company_name', '').strip().lower()
    team_skills_str = request.json.get('team_skills', '').strip().lower()
    
    # Parse existing team skills
    team_skills_set = set(re.findall(r'\b[a-z]{2,}\b', team_skills_str))

    if not job_desc:
        return jsonify({'error': 'Job Description cannot be empty.'}), 400
    if not CANDIDATES:
        return jsonify({'error': 'No candidate data loaded.'}), 500

    t0 = time.time()

    # 1. Parse JD features
    jd_features = extract_jd_features(job_desc)

    # Clean the job_desc to remove explicit YOE mentions before embedding
    clean_jd = re.sub(r'\d+\+?\s*(?:-|to|and)?\s*\d*\s*years?(?:\s+of\s+experience)?', '', job_desc, flags=re.IGNORECASE)

    # 2. Encode JD vector
    jd_vec = ai_model.encode([clean_jd], normalize_embeddings=True)

    # 3. Vectorised Semantic similarity
    semantic_scores = (CANDIDATE_EMBEDDINGS @ jd_vec[0]) * 100  # shape (N,)

    # 4. Compute dynamic logic for all candidates
    results = []
    for i, c in enumerate(CANDIDATES):
        sem = float(semantic_scores[i])
        cache = CANDIDATE_CACHE[i]
        
        pm = dynamic_profile_gate(c, jd_features)
        auth = dynamic_skill_authenticity(c, jd_features, cache['evidence'])
        am = dynamic_assessment_bonus(c, jd_features)
        
        raw = sem * pm * auth * cache['bm'] * cache['em'] * am
        
        sig = c.get('redrob_signals', {})
        
        # New Feature: Poach-ability (Flight Risk)
        days = days_since(sig.get('last_active_date', ''))
        otw = sig.get('open_to_work_flag', False)
        poachable = (days <= 14 and not otw)

        # New Feature: Missing Skills (Gap Analysis) & Team Complementarity
        candidate_skills_text = ' '.join(s['name'].lower() for s in c.get('skills', []))
        candidate_skills_set = set(re.findall(r'\b[a-z]{2,}\b', candidate_skills_text))
        
        missing_skills = [k for k in jd_features['keywords'] if k not in candidate_skills_text]
        
        # Team Complementarity (Puzzle Piece) - how many required skills do they have that the existing team lacks?
        adds_new_skills = 0
        if team_skills_set:
            required_skills_they_have = jd_features['keywords'].intersection(candidate_skills_set)
            new_skills_brought = required_skills_they_have - team_skills_set
            adds_new_skills = len(new_skills_brought)
            # Boost score for each unique skill they bring that the team doesn't have
            raw *= (1.0 + (adds_new_skills * 0.15))

        # New Feature: Team Multiplier (Glue Work)
        evidence_lower = cache['evidence'].lower()
        glue_words = ['mentor', 'lead', 'coach', 'train', 'agile', 'scrum', 'facilitate', 'cross-functional']
        is_team_multiplier = sum(1 for w in glue_words if w in evidence_lower) >= 2

        # New Feature: Career Personas & Rising Stars & Startup DNA
        gh = sig.get('github_activity_score', -1)
        history = c.get('career_history', [])
        yoe = c.get('profile', {}).get('years_of_experience', 0)
        
        company_dna = ""
        if len(history) >= 4 and yoe < 8:
            company_dna = "Startup Scrapper"
        elif len(history) <= 2 and yoe > 8:
            company_dna = "Enterprise Scaler"
        
        persona = "Specialist"
        if yoe <= 4 and gh > 80 and auth > 0.8:
            persona = "Rising Star"
        elif gh > 70:
            persona = "Open Source Contributor"
        elif len(history) >= 4 and all(days_since(h.get('start_date','')) < 730 for h in history[-2:]):
            persona = "Fast Tracker"
        elif len(history) > 0 and yoe > 5 and len(history) <= 2:
            persona = "Loyalist"

        # New Feature: Over-qualified Warning
        is_over_qualified = False
        max_y = jd_features['max_yoe']
        if max_y is not None and yoe > max_y + 3.0:
            is_over_qualified = True

        # Fetch core profile variables
        c_name = c.get('profile', {}).get('anonymized_name', 'there')
        title = c.get('profile', {}).get('current_title', 'N/A')
        
        # New Feature: Title Inflation Detector
        title_lower = title.lower()
        is_title_inflated = False
        # If they hold a very senior title but have < 6 years total experience
        if yoe < 6 and any(w in title_lower for w in ['cto', 'chief', 'vp', 'vice president', 'director', 'head of', 'principal']):
            is_title_inflated = True
            
        # New Feature: Mullet Candidate (Corporate on paper, Hacker in practice)
        is_mullet = False
        if gh > 80 and any(w in title_lower for w in ['manager', 'director', 'vp', 'head', 'enterprise', 'chief', 'president']):
            is_mullet = True

        # New Feature: Career Velocity (Rocketship)
        is_rocketship = False
        if yoe <= 5 and any(w in title_lower for w in ['senior', 'lead', 'staff', 'principal']) and not is_title_inflated:
            is_rocketship = True
            
        # New Feature: Anti-Network (0% overlap with team skills)
        is_anti_network = False
        if team_skills_set and len(candidate_skills_set.intersection(team_skills_set)) == 0:
            is_anti_network = True
            
        # New Feature: Silent Carrier (Glue work but non-managerial)
        is_silent_carrier = False
        if is_team_multiplier and yoe >= 4 and not any(w in title_lower for w in ['manager', 'director', 'vp', 'head', 'chief']):
            is_silent_carrier = True

        # Out-of-the-box Features: Peter Principle, Over-Employment, Wartime Survivor, Trough Survivor
        def parse_date(d_str):
            if not d_str or d_str.lower() == 'present': return date.today()
            try: return datetime.strptime(d_str, "%Y-%m-%d").date()
            except: return date.today()

        is_peter_principle = False
        is_overemployed = False
        is_wartime_survivor = False
        is_trough_survivor = False

        present_roles = []
        long_ic_roles = 0
        short_manager_roles = 0

        for h in history:
            s_date = parse_date(h.get('start_date', ''))
            e_date = parse_date(h.get('end_date', ''))
            duration_days = (e_date - s_date).days
            
            if duration_days >= 1825:
                is_wartime_survivor = True
            if duration_days >= 1460:
                is_trough_survivor = True
                
            if not h.get('end_date') or str(h.get('end_date')).lower() == 'present':
                present_roles.append(s_date)
                
            t_lower = h.get('title', '').lower()
            is_manager = any(w in t_lower for w in ['manager', 'director', 'vp', 'head', 'lead', 'chief'])
            
            if is_manager and 0 < duration_days < 500:
                short_manager_roles += 1
            if not is_manager and duration_days >= 1400:
                long_ic_roles += 1
                
        if long_ic_roles > 0 and short_manager_roles > 0:
            is_peter_principle = True
            
        if len(present_roles) >= 2:
            newest_start = max(present_roles)
            if (date.today() - newest_start).days > 60:
                is_overemployed = True

        # New Feature: Outreach Drafter
        c_title = title if title != 'N/A' else 'professional'
        company = history[0].get('company', 'your current company') if history else 'your current company'
        
        if persona == "Loyalist": hook = f"I noticed you've built an impressive track record at {company}."
        elif persona == "Fast Tracker": hook = f"I've been following your rapid career progression at {company}."
        elif persona == "Open Source Contributor": hook = f"I was really impressed by your technical depth and open source contributions."
        elif persona == "Rising Star": hook = f"Your trajectory is outstanding for someone early in their career."
        else: hook = f"Your background as a {c_title} caught my eye."
        
        email_draft = f"Hi {c_name},\n\n{hook} We are looking for someone with exactly your skill set to join us. I know you might not be actively looking, but I'd love to chat if you're open to a new challenge.\n\nBest,\n[Your Name]"

        # New Feature: Boomerang Candidate
        is_boomerang = False
        if company_name and len(history) > 1:
            for h in history[1:]:
                if company_name in h.get('company', '').lower():
                    is_boomerang = True
                    break

        # XAI Justification
        overlap = "Strong" if auth > 0.7 else ("Moderate" if auth > 0.4 else "Weak")
        xai = f"System matched {title} ({yoe} yrs exp). {overlap} skill intersection with {int(auth*100)}% verified authenticity."

        results.append({
            'candidate_id':   c['candidate_id'],
            'name':           c_name,
            'title':          title,
            'raw_score':      raw,
            'semantic_score': round(sem, 1),
            'authenticity_pct': int(auth * 100),
            'profile_mult':   round(pm, 3),
            'behavior_mult':  round(cache['bm'], 3),
            'open_to_work':   otw,
            'github_score':   gh,
            'last_active_days': days,
            'reasoning':      make_reasoning(c, sem, auth),
            'poachable':      poachable,
            'missing_skills': missing_skills[:3], # show up to 3 missing
            'career_persona': persona,
            'xai_justification': xai,
            'is_over_qualified': is_over_qualified,
            'email_draft':    email_draft,
            'is_boomerang':   is_boomerang,
            'company_dna':    company_dna,
            'is_team_multiplier': is_team_multiplier,
            'adds_new_skills': adds_new_skills,
            'is_title_inflated': is_title_inflated,
            'is_peter_principle': is_peter_principle,
            'is_overemployed': is_overemployed,
            'is_wartime_survivor': is_wartime_survivor,
            'is_trough_survivor': is_trough_survivor,
            'is_mullet': is_mullet,
            'is_anti_network': is_anti_network,
            'is_rocketship': is_rocketship,
            'is_silent_carrier': is_silent_carrier,
            'career_history_json': json.dumps(c.get('career_history', []))  # Used for AI Chat mock
        })

    # Normalise
    raw_scores = [r['raw_score'] for r in results]
    s_min, s_max = min(raw_scores), max(raw_scores)
    span = s_max - s_min if s_max != s_min else 1.0
    for r in results:
        r['score'] = round(max(0.0001, min(1.0000, (r['raw_score']-s_min)/span)), 4)

    # Sort + rank
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    for i, r in enumerate(results): r['rank'] = i + 1

    # Write CSV (top 100)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in results[:100]:
            w.writerow([r['candidate_id'], r['rank'], f"{r['score']:.4f}", r['reasoning']])

    elapsed = time.time() - t0
    print(f"[/rank] Scored {len(results):,} candidates for JD ({jd_features['min_yoe']}-{jd_features['max_yoe']}yoe) in {elapsed:.2f}s")

    return jsonify(results[:200])

@app.route('/download')
def download_csv():
    if not os.path.exists(OUTPUT_CSV):
        return jsonify({'error': 'CSV not generated yet. Run a ranking first.'}), 404
    return send_file(OUTPUT_CSV, as_attachment=True, download_name='team_submission.csv')

@app.route('/status')
def status():
    return jsonify({
        'candidates_loaded': len(CANDIDATES),
        'embeddings_cached': os.path.exists(EMBED_CACHE),
        'csv_ready': os.path.exists(OUTPUT_CSV),
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
