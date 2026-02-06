import logging
import ssl
import concurrent.futures
from typing import Iterable, Optional, Tuple, Set
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords

# Configure root logger (users can override externally)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Translator:
    """Smart translator with English detection heuristics and timeout safety."""

    def __init__(
        self,
        target_lang: str = 'en',
        timeout_sec: int = 10,
        max_text_length: int = 4500,
        english_stopwords_threshold: int = 2
    ):
        self.target_lang = target_lang
        self.timeout_sec = timeout_sec
        self.max_text_length = max_text_length
        self.stopwords_threshold = english_stopwords_threshold
        self._stopwords: Set[str] = set()
        self._setup_nltk()

    def _setup_nltk(self) -> None:
        """Handle NLTK SSL quirks and load stopwords."""
        try:
            _create_unverified_https_context = ssl._create_unverified_https_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        try:
            nltk.download('stopwords', quiet=True)
            self._stopwords = set(w.lower() for w in stopwords.words('english'))
        except Exception as e:
            logger.warning(f"⚠️  Failed to load stopwords: {e}")
            self._stopwords = set()

    def detect_language(self, text: str) -> Optional[str]:
        """Detect primary language; returns None on failure."""
        try:
            return detect(text.strip()) or None
        except LangDetectException:
            return None

    def _is_likely_english(self, text: str) -> bool:
        """Heuristic check: stopwords + langdetect confirmation."""
        words = set(w.lower() for w in text.split())
        if len(words & self._stopwords) >= self.stopwords_threshold:
            detected = self.detect_language(text)
            return detected is None or detected == 'en'
        return False

    def translate(self, text: str, source_lang: Optional[str] = None) -> Tuple[str, bool]:
        """
        Translate text to target language.
        Returns:
            (translated_text, was_translated)
        """
        text = text.strip()
        if not text:
            return text, False

        # Skip translation if explicitly English or heuristically detected
        if source_lang == 'en' or self._is_likely_english(text):
            logger.debug("✅ Skipping translation: text appears to be English")
            return text, False

        logger.info(f"🌐 Translating: '{text[:50]}...' (source={source_lang or 'auto'})")
        try:
            translator = GoogleTranslator(
                source=source_lang or 'auto',
                target=self.target_lang,
                timeout=8
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(translator.translate, text[:self.max_text_length])
                result = future.result(timeout=self.timeout_sec)

            logger.info(f"✅ Translated: '{result[:50]}...'")
            return result, True

        except concurrent.futures.TimeoutError:
            logger.error(f"⏰ Translation timed out after {self.timeout_sec}s")
        except Exception as e:
            logger.error(f"❌ Translation failed: {type(e).__name__}: {str(e)[:100]}")

        return text, False
    
    def translate_skills(self, skills: Iterable[str], chunk_size: int = 100) -> Set[str]:
        """Batch-translate a collection of skills (e.g., resume keywords)."""
        result: Set[str] = set()
        batch: list[str] = []
        batch_len = 0

        for skill in map(str.strip, skills):
            if not skill:
                continue
            if not self.detect_language(skill):  # English → keep as-is
                result.add(skill)
                continue

            batch.append(skill)
            batch_len += len(skill)

            if batch_len >= chunk_size or len(batch) >= 10:
                translated, _ = self.translate(" ; ".join(batch))
                result.update(s.strip() for s in translated.split(" ; ") if s.strip())
                batch.clear()
                batch_len = 0

        if batch:
            translated, _ = self.translate(" ; ".join(batch))
            result.update(s.strip() for s in translated.split(" ; ") if s.strip())

        return result


# Module-level convenience instance
_default_translator = Translator()


def translate(text: str, source_lang: Optional[str] = None) -> Tuple[str, bool]:
    """Convenient module-level translation function."""
    return _default_translator.translate(text, source_lang)


def detect_language(text: str) -> Optional[str]:
    """Convenient module-level language detection."""
    return _default_translator.detect_language(text)


if __name__ == "__main__":
    print("✓ Translator module loaded. Ready for resume/content processing.")
    print("Usage examples:")
    print("  from translator import translate")
    print("  text, was_translated = translate('Bonjour le monde')")