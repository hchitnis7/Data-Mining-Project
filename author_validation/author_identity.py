"""Author identity normalization. Exact matching avoids speculative initial expansion."""
import re
import unicodedata


def normalize_name(value):
    value = unicodedata.normalize('NFKC', value or '').casefold()
    return ' '.join(re.findall(r'[^\W_]+', value, flags=re.UNICODE))


def full_author_key(author):
    parts = list(author) + ['', '', '']
    last, given, suffix = map(normalize_name, parts[:3])
    # Initial-only and missing names are deliberately unresolved.
    if not last or not any(len(token) > 1 for token in given.split()):
        return ''
    return '|'.join((last, given, suffix))
