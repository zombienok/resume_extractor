import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable
from collections import defaultdict
import pickle
import logging

import torch
from sentence_transformers import util

from src.extract_skills import ModelManager, SkillExtractor, SkillMatcher  # Previous module

# Your parser (included inline for cohesion)
import PyPDF2
from docx import Document
from io import BytesIO

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF/DOCX/TXT files"""
    try:
        if filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        
        elif filename.lower().endswith('.docx'):
            doc = Document(BytesIO(file_bytes))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        
        elif filename.lower().endswith('.txt'):
            return file_bytes.decode('utf-8', errors='ignore')
        
        raise ValueError(f"Unsupported file type: {filename}")
    
    except Exception as e:
        raise ValueError(f"Text extraction failed for {filename}: {str(e)}")


logger = logging.getLogger(__name__)


class BatchResumeProcessor:
    """
    Batch processor for resume skill extraction with:
    • Multi-format support (PDF/DOCX/TXT)
    • Single-pass chunk encoding (deduplicated)
    • Disk caching of embeddings
    • Zero per-resume model loading after init
    """
    
    # Parser version – increment if extract_text() logic changes to invalidate cache
    _PARSER_VERSION = "1.0"
    
    def __init__(
        self,
        skills_db: List[str],
        cache_dir: str = ".skill_cache",
        threshold: float = 0.55,
    ):
        if not skills_db:
            raise ValueError("skills_db cannot be empty")
        
        self.skills_db = skills_db
        self.threshold = threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Core components
        self.matcher = SkillMatcher(skills_db, threshold)
        self.resume_texts: Dict[str, str] = {}          # resume_id → raw text
        self.resume_chunks: Dict[str, List[str]] = {}   # resume_id → [chunks]
        self.all_unique_chunks: Set[str] = set()
        self.chunk_to_embedding: Dict[str, torch.Tensor] = {}
        self.chunk_to_resumes: Dict[str, Set[str]] = defaultdict(set)
    
    def _get_cache_key(self) -> str:
        """Generate cache key incorporating parser version + model + chunks."""
        model = ModelManager.get_sentence_model()
        model_name = getattr(model, 'model_name', str(model))
        chunks_hash = hashlib.md5(
            json.dumps(sorted(self.all_unique_chunks), ensure_ascii=False).encode()
        ).hexdigest()
        return f"{model_name.replace('/', '_')}_v{self._PARSER_VERSION}_{chunks_hash}.pkl"
    
    def _load_cache(self) -> bool:
        cache_file = self.cache_dir / self._get_cache_key()
        if not cache_file.exists():
            return False
        
        try:
            logger.info(f"Loading cached embeddings: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            if not isinstance(cache, dict) or 'embeddings' not in cache:
                return False
            
            self.chunk_to_embedding = cache['embeddings']
            logger.info(f"✓ Loaded {len(self.chunk_to_embedding)} cached embeddings")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False
    
    def _save_cache(self) -> None:
        cache_file = self.cache_dir / self._get_cache_key()
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.chunk_to_embedding,
                    'chunks': list(self.all_unique_chunks),
                    'parser_version': self._PARSER_VERSION,
                    'timestamp': time.time()
                }, f)
            logger.info(f"✓ Saved cache ({len(self.chunk_to_embedding)} embeddings)")
        except Exception as e:
            logger.warning(f"Cache save failed (non-critical): {e}")
    
    # === Resume ingestion methods ===
    
    def add_resume(self, resume_id: str, file_bytes: bytes, filename: str) -> None:
        """
        Add a single resume from raw bytes (PDF/DOCX/TXT).
        
        Args:
            resume_id: Unique identifier (e.g., "john_doe_2024")
            file_bytes: Raw file content
            filename: Original filename (used to detect format)
        """
        try:
            text = extract_text(file_bytes, filename)
            if not text.strip():
                logger.warning(f"Skipping empty resume: {resume_id}")
                return
            
            self.resume_texts[resume_id] = text
            chunks = SkillExtractor.extract_chunks(text)
            
            if chunks:
                self.resume_chunks[resume_id] = chunks
                for chunk in chunks:
                    self.all_unique_chunks.add(chunk)
                    self.chunk_to_resumes[chunk].add(resume_id)
            else:
                logger.debug(f"No skill chunks found in {resume_id}")
        
        except Exception as e:
            logger.error(f"Failed to process {resume_id} ({filename}): {e}")
            raise
    
    def add_resume_from_path(self, file_path: str, resume_id: Optional[str] = None) -> None:
        """Add resume from filesystem path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'rb') as f:
            file_bytes = f.read()
        
        rid = resume_id or path.stem
        self.add_resume(rid, file_bytes, path.name)
    
    def add_resumes_from_folder(
        self,
        folder_path: str,
        resume_id_fn: Optional[Callable[[Path], str]] = None
    ) -> None:
        """
        Add all supported resumes from a folder.
        
        Args:
            folder_path: Directory containing resumes
            resume_id_fn: Optional function Path → str for custom IDs
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        supported = list(folder.glob("*.pdf")) + list(folder.glob("*.docx")) + list(folder.glob("*.txt"))
        if not supported:
            logger.warning(f"No supported files (.pdf/.docx/.txt) in {folder}")
            return
        
        logger.info(f"Processing {len(supported)} resumes from {folder}")
        for path in supported:
            rid = resume_id_fn(path) if resume_id_fn else path.stem
            self.add_resume_from_path(str(path), rid)
        
        logger.info(
            f"✓ Extracted {len(self.all_unique_chunks)} unique chunks "
            f"from {len(self.resume_chunks)} resumes"
        )
    
    # === Embedding & matching ===
    
    def encode_chunks(self, use_cache: bool = True, batch_size: int = 32) -> None:
        """Batch-encode all unique chunks (deduplicated)."""
        if not self.all_unique_chunks:
            raise ValueError("No chunks extracted. Add resumes first.")
        
        if use_cache and self._load_cache():
            missing = [c for c in self.all_unique_chunks if c not in self.chunk_to_embedding]
            if not missing:
                return
            logger.info(f"Cache missing {len(missing)} chunks – encoding remainder")
            chunks_to_encode = missing
        else:
            chunks_to_encode = list(self.all_unique_chunks)
        
        if chunks_to_encode:
            logger.info(f"Encoding {len(chunks_to_encode)} chunks in batch...")
            model = ModelManager.get_sentence_model()
            embeddings = model.encode(
                chunks_to_encode,
                convert_to_tensor=True,
                show_progress_bar=len(chunks_to_encode) > 50,
                batch_size=batch_size
            )
            
            for chunk, emb in zip(chunks_to_encode, embeddings):
                self.chunk_to_embedding[chunk] = emb
            
            self._save_cache()
        else:
            logger.info("✓ All chunks already encoded")
    
    def match_all_resumes(self, top_k: Optional[int] = None) -> Dict[str, List[Tuple[str, float]]]:
        """Match all added resumes using precomputed embeddings."""
        if not self.chunk_to_embedding:
            raise RuntimeError("Call encode_chunks() before matching.")
        
        logger.info("Matching resumes against skill database...")
        results: Dict[str, List[Tuple[str, float]]] = {}
        skill_embs = self.matcher._skill_embeddings
        
        # Precompute global best chunk per skill (across ALL resumes)
        chunk_list = list(self.chunk_to_embedding.keys())
        chunk_tensors = torch.stack([self.chunk_to_embedding[c] for c in chunk_list])
        sim_matrix = util.cos_sim(chunk_tensors, skill_embs)  # [n_chunks, n_skills]
        best_sims = sim_matrix.max(dim=0).values.cpu().numpy()
        
        # For each resume: collect skills where its chunks exceed threshold
        for resume_id, chunks in self.resume_chunks.items():
            resume_skills = []
            seen = set()
            
            # Get embeddings for this resume's chunks
            valid_chunks = [c for c in chunks if c in self.chunk_to_embedding]
            if not valid_chunks:
                results[resume_id] = []
                continue
            
            resume_embs = torch.stack([self.chunk_to_embedding[c] for c in valid_chunks])
            local_sims = util.cos_sim(resume_embs, skill_embs)  # [n_resume_chunks, n_skills]
            chunk_best = local_sims.max(dim=0).values.cpu().numpy()  # Best per skill in this resume
            
            for skill, sim in zip(self.skills_db, chunk_best):
                if sim > self.threshold and skill not in seen:
                    resume_skills.append((skill, float(sim)))
                    seen.add(skill)
            
            resume_skills.sort(key=lambda x: x[1], reverse=True)
            results[resume_id] = resume_skills[:top_k] if top_k else resume_skills
        
        logger.info(f"✓ Matched skills for {len(results)} resumes")
        return results
    
    def get_skill_frequency(self) -> Dict[str, int]:
        """Count skill occurrences across all resumes."""
        results = self.match_all_resumes()
        freq = defaultdict(int)
        for matches in results.values():
            for skill, _ in matches:
                freq[skill] += 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    
    def clear(self) -> None:
        """Reset processor state (keep matcher/cache config)."""
        self.resume_texts.clear()
        self.resume_chunks.clear()
        self.all_unique_chunks.clear()
        self.chunk_to_embedding.clear()
        self.chunk_to_resumes.clear()
        logger.info("Processor state cleared")
    
    def clear_cache(self) -> None:
        """Delete all embedding caches from disk."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        logger.info(f"Cleared cache directory: {self.cache_dir}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    skills = [
        "Python", "JavaScript", "React", "TensorFlow", "PyTorch", "Docker",
        "Kubernetes", "AWS", "SQL", "Git", "Machine Learning", "NLP",
        "Computer Vision", "Data Analysis", "REST APIs", "CI/CD"
    ]
    
    processor = BatchResumeProcessor(skills_db=skills, threshold=0.6)
    
    # Method 1: From folder (auto-detects PDF/DOCX/TXT)
    processor.add_resumes_from_folder("./resumes")
    
    # Method 2: Programmatic (e.g., from web upload)
    # with open("resume.pdf", "rb") as f:
    #     processor.add_resume("user_123", f.read(), "resume.pdf")
    
    # Encode once for ALL unique chunks
    processor.encode_chunks(use_cache=True)
    
    # Match all instantly
    results = processor.match_all_resumes(top_k=5)
    
    for rid, skills in results.items():
        print(f"\n📄 {rid}:")
        for skill, score in skills:
            print(f"  • {skill}: {score:.3f}")
    
    print("\n📊 Top skills across resumes:")
    for skill, count in list(processor.get_skill_frequency().items())[:8]:
        print(f"  {skill}: {count} resumes")