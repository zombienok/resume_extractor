"""
test.py — uv-friendly standalone tests
Run with: uv run python test.py
"""

from translator import detect_language, translate_to_english
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def test_detect_language():
    print("🔍 Language Detection Tests")
    cases = [
        ("Hello world", None, "English → skip"),
        ("Привет мир", "ru", "Russian"),
        ("Hola mundo", "es", "Spanish"),
        ("", None, "Empty text"),
    ]
    for text, expected, label in cases:
        result = detect_language(text)
        mark = "✓" if result == expected else "✗"
        print(f"  {mark} '{label}': '{text}' → {result!r} (expected {expected!r})")

def test_translation():
    print("\n🌍 Translation Tests")
    cases = [
        ("Senior Python Developer", "English (no translation)"),
        ("Старший разработчик Python", "Russian"),
        ("Entwickler für KI", "German"),
        ("", "Empty string"),
    ]
    for text, label in cases:
        translated, was_translated = translate_to_english(text)
        mark = "✓" if was_translated or text == translated else "?"
        print(f"  {mark} '{label}'")
        print(f"     IN : {text!r}")
        print(f"     OUT: {translated!r} (translated={was_translated})")

def test_stopwords_heuristic():
    print("\n🧠 Stopwords Heuristic Test")
    # Text with English stopwords should skip translation
    text = "The machine learning model is working well"
    translated, was_translated = translate_to_english(text)
    print(f"  Input with stopwords: {text!r}")
    print(f"  Skipped translation: {not was_translated} {'✓' if not was_translated else '✗'}")
    return 

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 uv TRANSLATOR TESTS")
    print("=" * 60)
    start_time = overall_time = time.time()
    test_detect_language()
    print(f"\n⏱️  Language detection tests completed in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()
    test_translation()
    print(f"\n⏱️  Translation tests completed in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()
    test_stopwords_heuristic()
    print(f"\n⏱️  Stopwords heuristic test completed in {time.time() - start_time:.2f} seconds.")

    
    print("\n" + "=" * 60)
    print("✅ Tests completed in {:.2f} seconds.".format(time.time() - overall_time))
    print("=" * 60)