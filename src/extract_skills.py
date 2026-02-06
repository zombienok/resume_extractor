import re
import spacy
from typing import List, Tuple, Set
from sentence_transformers import SentenceTransformer, util
import logging
import time


logging.basicConfig(level=logging.INFO, format='%(message)s')

# Lazy-loaded models
_nlp = None
_model = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Install it with:\n"
                "python -m spacy download en_core_web_sm"
            )
    return _nlp

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model



# Precompute skill embeddings once
_SKILL_EMBEDDINGS = None

def _get_skill_embeddings(skills_db: List[str]):
    global _SKILL_EMBEDDINGS
    if _SKILL_EMBEDDINGS is None:
        model = get_model()
        _SKILL_EMBEDDINGS = model.encode(skills_db, convert_to_tensor=True)
    return _SKILL_EMBEDDINGS


def extract_skill_chunks(text: str) -> List[str]:
    """
    Extract skill-like text chunks using spaCy:
    - Named entities (ORG, PRODUCT, SKILL-like)
    - Noun chunks containing technical terms
    - Short phrases (2-4 words) likely to be skills/tools
    """
    nlp = get_nlp()
    doc = nlp(text)
    
    chunks = set()
    
    # ALL noun chunks (1-4 words) — no keyword filtering
    for chunk in doc.noun_chunks:
        to_add = re.sub(r'[^a-zA-Z0-9 ]', '', chunk.text.strip().replace('\n', ' '))
        words = to_add.split()
        if 1 <= len(words) <= 4 and len(chunk.text) > 2:
            chunks.add(to_add)

    print(chunks)
    return list(chunks)


def match_skills(resume_text: str, skills_db: List[str], threshold: float = 0.55) -> Set[Tuple[str, float]]:
    """
    Match skills by:
    1. Extracting skill-like chunks from resume
    2. Semantically matching each chunk → skill DB
    3. Aggregating best match per skill
    """
    if not resume_text or not resume_text.strip():
        return []
    all_skills = set()
    # Step 1: Extract candidate chunks
    chunks = extract_skill_chunks(resume_text)
    if not chunks:
        return []
    
    # Step 2: Encode chunks and skills
    model = get_model()
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    skill_embs = _get_skill_embeddings(skills_db)
    
    # Step 3: Compute similarity matrix (chunks × skills)
    sim_matrix = util.cos_sim(chunk_embs, skill_embs)  # Shape: [n_chunks, n_skills]
    
    # Step 4: For each skill, take max similarity across chunks
    best_sims = sim_matrix.max(dim=0).values.cpu().numpy()
    
    matched_skills = list()
    # Step 5: Filter & sort
    for skill, sim in zip(skills_db, best_sims):
        if sim > threshold and skill not in all_skills: # Higher threshold for chunk-level matching
            all_skills.add(skill)
            matched_skills.append((skill, sim))
    return matched_skills

    

# # Example usage
# if __name__ == "__main__":
#     start_time = time.time()
#     resume = """
#     Senior Python Developer with 5+ years of experience in machine learning, deep learning, and NLP.
#     """
#     need_to_translate = detect_language(resume)
#     if need_to_translate:
#         logging.info("🌐 Detected non-English resume. Translating to English for skill extraction...")
#         translated_resume, was_translated = translate_to_english(resume, detect_language(resume))
#     results = match_skills(translated_resume) if translated_resume else match_skills(resume)
#     logging.info(f"Matching took {time.time() - start_time:.2f} seconds")
#     print("Top skill matches:")
#     for skill, score in results:
#         print(f"  {skill}: {score:.3f}")