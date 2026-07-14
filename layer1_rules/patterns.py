# Placeholder for Layer 1 regex and pattern rules.
"""
Regex checks: PII + known jailbreak/injection trigger phrases.
Fast, no ML. Will miss anything not on this list — that gap is what
Layer 2 (the trained classifier) exists to cover. Don't expect high recall here.
"""
import re

CREDIT_CARD_RE = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')

def luhn_check(card_number):
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if len(digits) < 13:
        return False
    checksum = 0
    digits.reverse()
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def detect_pii(text):
    hits = []
    for match in CREDIT_CARD_RE.finditer(text):
        if luhn_check(match.group()):
            hits.append(("credit_card", match.group()))
    if SSN_RE.search(text):
        hits.append(("ssn", SSN_RE.search(text).group()))
    if EMAIL_RE.search(text):
        hits.append(("email", EMAIL_RE.search(text).group()))
    return hits

TRIGGER_PHRASES = [
    # direct instruction override
    r'ignore (all|any|previous|prior|the above)( .*?)? instructions',
    r'disregard (your|all|any|the) (guidelines|rules|programming|instructions)',
    r'forget (your|all|previous) (instructions|rules|programming)',

    # role-hijack framing
    r'you (are|will be) (now |about to be )?(no longer|going to act as|immers\w* )',
    r'act as (an? )?(unrestricted|uncensored|amoral|unfiltered)',
    r'pretend (you|to) (have no|are not bound by) (restrictions|rules|filters)',
    r'from (now on|this moment)( onward)?,? you (are|will)',

    # named jailbreak personas
    r'\bDAN\b',
    r'do anything now',
    r'developer mode',
    r'\bAGI\b.*(zenith|unrestricted|no limits)',

    # uncensored-persona descriptors
    r'\b(amoral|unfiltered|uncensored)\b.*\b(AI|model|assistant)\b',
    r'does not (care|give a) .*(ethic|moral|rule)',

    # meta / self-referential
    r'reveal your (instructions|prompt|system message)',
    r'jailbreak',
]
TRIGGER_RE = re.compile('|'.join(TRIGGER_PHRASES), re.IGNORECASE)

def detect_trigger_phrases(text):
    return TRIGGER_RE.findall(text)

def check_layer1(text):
    pii_hits = detect_pii(text)
    trigger_hits = detect_trigger_phrases(text)
    return {
        "blocked": bool(pii_hits or trigger_hits),
        "pii_hits": pii_hits,
        "trigger_hits": trigger_hits,
    }