from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords
import ssl
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
# Fix NLTK SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def detect_language(text: str) -> str | bool:
    """Detect primary language of text (prioritize non-English detection)"""
    try:
        lang = detect(text.strip())
        return lang if lang != 'en' else False
    except LangDetectException:
        return False

def translate_to_english(text: str, source_lang: str = None) -> tuple[str, bool]:
    """Translate text to English. Returns (translated_text, was_translated)"""
    if not text.strip():
        logging.debug("EmptyEntries: Skipping translation")
        return text, False

    # === NEW: Multi-layer skip check for English text ===
    should_skip = False
    
    # Layer 1: Explicit English source language
    if source_lang == 'en':
        should_skip = True
        logging.debug("✅ Skipping: source_lang explicitly 'en'")
    
    # Layer 2: Stopwords heuristic (lower threshold + case-insensitive)
    elif len(set(text.lower().split()) & STOPWORDS) >= 2:  # ≥2 stopwords (was >3)
        # Layer 3: Verify with language detection
        detected_lang = detect_language(text)
        if detected_lang is None or detected_lang == 'en':
            should_skip = True
            logging.debug(f"✅ Skipping: stopwords + langdetect confirm English (detected='{detected_lang}')")
    
    if should_skip:
        return text, False
    # =====================================================

    # Proceed to translation only if NOT skipped
    logging.info(f"🌐 TRANSLATING: '{text[:50]}...' (lang={source_lang or 'auto'})")
    
    try:
        translator = GoogleTranslator(
            source=source_lang or 'auto',
            target='en',
            timeout=8  # Stricter timeout
        )
        # CRITICAL: Wrap in hard timeout to prevent hangs
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(translator.translate, text[:4500])
            translated = future.result(timeout=10)  # Hard 10s cutoff
        
        logging.info(f"✅ Translation successful: '{translated[:50]}...'")
        return translated, True
        
    except concurrent.futures.TimeoutError:
        logging.error("⏰ Translation TIMED OUT after 10 seconds (network issue?)")
        return text, False
    except Exception as e:
        logging.error(f"❌ Translation failed: {type(e).__name__}: {str(e)[:100]}")
        return text, False


print("Translator module loaded. Ready to detect and translate resume content.")