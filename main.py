#!/usr/bin/env python3
"""Batch resume skill extractor with auto-translation."""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.extract_skills import get_default_matcher, SkillExtractor
from src.precompute import BatchResumeProcessor, extract_text
from src.translator import Translator


import time


# Patch processor to accept pre-translated text (avoids re-extraction)
def add_resume_from_text(self, resume_id: str, text: str):
    if not text.strip():
        logging.warning(f"Skipping empty resume: {resume_id}")
        return
    self.resume_texts[resume_id] = text
    chunks = SkillExtractor.extract_chunks(text)
    if chunks:
        self.resume_chunks[resume_id] = chunks
        for chunk in chunks:
            self.all_unique_chunks.add(chunk)
            self.chunk_to_resumes[chunk].add(resume_id)
    else:
        logging.debug(f"No skill chunks found in {resume_id}")

def main():
    parser = argparse.ArgumentParser(description="Extract skills from resumes in a folder")
    parser.add_argument("folder", type=str, help="Path to folder containing resumes (.pdf/.docx/.txt)")
    parser.add_argument("-o", "--output", default="results.txt", help="Output file (default: results.txt)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    translator = Translator()
    matcher = get_default_matcher()
    
    # Initialize batch processor with default skills
    processor = BatchResumeProcessor(
        skills_db=matcher.skills_db,
        threshold=matcher.threshold,
        cache_dir=".skill_cache"
    )
    processor.add_resume_from_text = lambda rid, txt: add_resume_from_text(processor, rid, txt)

    # Process all resumes: extract → translate → add
    folder = Path(args.folder)
    for path in folder.glob("*"):
        if path.suffix.lower() not in (".pdf", ".docx", ".txt"):
            continue
        
        try:
            # Extract raw text
            with open(path, "rb") as f:
                raw_text = extract_text(f.read(), path.name)
            
            # Translate if non-English
            lang = translator.detect_language(raw_text[:500])  # Sample first 500 chars
            if lang and lang != "en":
                logging.info(f"🌍 Translating {path.name} (detected: {lang})")
                raw_text, _ = translator.translate(raw_text)
            
            processor.add_resume_from_text(path.stem, raw_text)
            logging.info(f"✓ Processed: {path.name}")
        
        except Exception as e:
            logging.error(f"❌ Failed {path.name}: {e}")

    if not processor.resume_chunks:
        logging.error("No valid resumes processed. Exiting.")
        return

    # SINGLE PASS: encode all unique chunks across all resumes
    processor.encode_chunks(use_cache=True, batch_size=32)
    
    # Match against precomputed skill embeddings (no re-encoding)
    results: Dict[str, List[Tuple[str, float]]] = processor.match_all_resumes(top_k=10)

    # Write results
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("RESUME SKILL MATCHING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        for rid, skills in sorted(results.items()):
            f.write(f"📄 {rid}\n")
            if not skills:
                f.write("  (no skills matched above threshold)\n")
            for skill, score in skills:
                f.write(f"  • {skill}: {score:.3f}\n")
            f.write("\n")
        
        # Skill frequency summary
        f.write("\n" + "=" * 60 + "\n")
        f.write("TOP SKILLS ACROSS ALL RESUMES\n")
        f.write("=" * 60 + "\n")
        freq = processor.get_skill_frequency()
        for i, (skill, count) in enumerate(list(freq.items())[:15], 1):
            f.write(f"{i:2d}. {skill:30s} ({count} resumes)\n")
    
    logging.info(f"\n✓ Results written to {args.output}")
    logging.info(f"  Processed {len(results)} resumes with {len(processor.all_unique_chunks)} unique chunks")

if __name__ == "__main__":
    start_time = time.time()
    main()
    logging.info(f"\n⏱️  Total execution time: {time.time() - start_time:.2f} seconds")