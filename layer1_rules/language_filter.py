"""
Uses actual language detection instead of a crude character-ratio guess.
"""
try:
    from langdetect import detect, LangDetectException
except ImportError:  # pragma: no cover - optional dependency
    detect = None

    class LangDetectException(Exception):
        pass


def is_likely_english(text):
    if len(text.strip()) < 5:
        return True  # too short to reliably detect, don't drop it
    if detect is None:
        return True  # dependency unavailable, don't drop it silently
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return True  # detection failed, don't drop it silently