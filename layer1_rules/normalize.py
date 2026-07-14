# Placeholder for Layer 1 normalization logic.
"""
Cleans text before pattern matching so simple obfuscation doesn't slip through.
"""
import re
import base64
import unicodedata

def strip_zero_width(text):
    for ch in ['\u200b', '\u200c', '\u200d', '\ufeff']:
        text = text.replace(ch, '')
    return text

def fix_homoglyphs(text):
    return unicodedata.normalize('NFKC', text)

def try_decode_base64_chunks(text):
    candidates = re.findall(r'[A-Za-z0-9+/]{16,}={0,2}', text)
    decoded_parts = []
    for c in candidates:
        try:
            decoded = base64.b64decode(c, validate=True).decode('utf-8')
            if decoded.isprintable():
                decoded_parts.append(decoded)
        except Exception:
            continue
    if decoded_parts:
        return text + " " + " ".join(decoded_parts)
    return text

def normalize(text):
    text = strip_zero_width(text)
    text = fix_homoglyphs(text)
    text = try_decode_base64_chunks(text)
    return text.lower()