"""Deterministic Indonesian text normalization for the TTS frontend.

Turns noisy LLM output (markdown, symbols, raw numbers) into clean spoken
Indonesian with zero token cost, fitting the per-sentence streaming TTS path.
Ports the rule spec in `src/prompts/text-to-speech.txt`; order matters:
markdown/symbol cleanup → number/roman normalization → acronym spelling →
lowercase → whitespace.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

try:
    from num2words import num2words as _num2words

    def _spell_number(value: int) -> str:
        return _num2words(value, lang="id")

    _HAS_NUM2WORDS = True
except Exception:  # pragma: no cover - exercised only when dep missing
    logger.warning("num2words unavailable; numbers will not be expanded for TTS")
    _HAS_NUM2WORDS = False

    def _spell_number(value: int) -> str:
        return str(value)


# Indonesian letter pronunciation for spelling out acronyms (PTN -> pe te en).
_LETTER_SOUNDS = {
    "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "ef",
    "G": "ge", "H": "ha", "I": "i", "J": "je", "K": "ka", "L": "el",
    "M": "em", "N": "en", "O": "o", "P": "pe", "Q": "ki", "R": "er",
    "S": "es", "T": "te", "U": "u", "V": "fe", "W": "we", "X": "eks",
    "Y": "ye", "Z": "zet",
}

# All-caps tokens that should be read as ordinary words, not spelled out.
_ACRONYM_WORD_WHITELIST = {"UDINUS"}

# Roman numerals only converted after these keywords, to avoid mangling stray
# capital letters that merely look Roman (e.g. a lone "I" or "V").
_ROMAN_KEYWORDS = r"(?:bab|semester|jilid|kelas|abad|seri|tingkat|level)"
_ROMAN_RE = re.compile(
    rf"\b({_ROMAN_KEYWORDS})\s+(M{{0,3}}(?:CM|CD|D?C{{0,3}})(?:XC|XL|L?X{{0,3}})(?:IX|IV|V?I{{0,3}}))\b",
    re.IGNORECASE,
)

_HTML_ENTITIES = {
    "&amp;": " dan ",
    "&lt;": " kurang dari ",
    "&gt;": " lebih dari ",
    "&quot;": '"',
    "&#39;": "'",
    "&nbsp;": " ",
}


def _roman_to_int(roman: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for char in reversed(roman.upper()):
        cur = values.get(char, 0)
        total += -cur if cur < prev else cur
        prev = max(prev, cur)
    return total


def _strip_markdown(text: str) -> str:
    # Fenced and inline code: drop the fence/backticks, keep the text.
    text = re.sub(r"```[a-zA-Z0-9]*\n?", " ", text)
    text = text.replace("`", "")
    # Images ![alt](url) -> alt ; links [text](url) -> text.
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Headings, blockquotes, list bullets at line starts.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}\d+\.\s+", "", text)
    # Horizontal rules.
    text = re.sub(r"(?m)^\s{0,3}([-*_])\1{2,}\s*$", "", text)
    # Bold/italic markers.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    return text


def _normalize_punctuation(text: str) -> str:
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "–": "-", "…": ".",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{3,}", ".", text)
    return text


def _spell_letters(token: str) -> str:
    return " ".join(_LETTER_SOUNDS.get(ch, ch) for ch in token.upper())


def _expand_emails_and_urls(text: str) -> str:
    def _email(match: re.Match) -> str:
        local, domain = match.group(1), match.group(2)
        spoken_domain = domain.replace(".", " dot ")
        return f"{local} at {spoken_domain}"

    text = re.sub(r"\b([\w.+-]+)@([\w.-]+\.\w+)\b", _email, text)

    # Domains / URLs: word.word(.tld) -> "word dot word dot tld". Require an
    # alphabetic final segment (TLD) so numeric groups like "25.000" (thousands
    # separators) are left for the number normalizer, not read as a domain.
    def _url(match: re.Match) -> str:
        return match.group(0).replace(".", " dot ")

    text = re.sub(r"\b[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b", _url, text)
    return text


def _expand_numbers(text: str) -> str:
    # Currency: Rp 25.000 -> "dua puluh lima ribu rupiah".
    def _currency(match: re.Match) -> str:
        digits = match.group(1).replace(".", "").replace(",", "")
        try:
            return f"{_spell_number(int(digits))} rupiah"
        except ValueError:
            return match.group(0)

    text = re.sub(r"\bRp\.?\s?(\d[\d.,]*)", _currency, text, flags=re.IGNORECASE)

    # Percent: 10% / 10,5% -> "sepuluh persen" / "sepuluh koma lima persen".
    text = re.sub(
        r"(\d[\d.,]*)\s*%",
        lambda m: f"{_expand_number_token(m.group(1))} persen",
        text,
    )

    # Multiplication / dimension: 4x6 / 1920×1080 -> "... kali ...".
    text = re.sub(
        r"(\d+)\s*[x×]\s*(\d+)",
        lambda m: f"{_spell_number(int(m.group(1)))} kali {_spell_number(int(m.group(2)))}",
        text,
    )

    # Numeric range: 6-8 -> "enam sampai delapan".
    text = re.sub(
        r"(\d+)\s*-\s*(\d+)",
        lambda m: f"{_spell_number(int(m.group(1)))} sampai {_spell_number(int(m.group(2)))}",
        text,
    )

    # Degree / level codes: S1 -> "es satu", D3 -> "de tiga".
    text = re.sub(
        r"\b([A-Z])(\d)\b",
        lambda m: f"{_spell_letters(m.group(1))} {_spell_number(int(m.group(2)))}",
        text,
    )

    # Remaining standalone numbers (thousands-grouped, decimal, or plain).
    text = re.sub(r"\d[\d.,]*", lambda m: _expand_number_token(m.group(0)), text)
    return text


def _expand_number_token(token: str) -> str:
    token = token.strip().strip(".,")
    if not token:
        return token
    # Decimal (Indonesian comma): 3,14 -> "tiga koma satu empat".
    if "," in token:
        whole, _, frac = token.partition(",")
        whole = whole.replace(".", "")
        try:
            words = _spell_number(int(whole)) if whole else "nol"
        except ValueError:
            return token
        digits = " ".join(_spell_number(int(d)) for d in frac if d.isdigit())
        return f"{words} koma {digits}".strip()
    # Integer, possibly with "." thousands separators.
    cleaned = token.replace(".", "")
    if not cleaned.isdigit():
        return token
    try:
        return _spell_number(int(cleaned))
    except ValueError:
        return token


def _expand_roman(text: str) -> str:
    return _ROMAN_RE.sub(
        lambda m: f"{m.group(1)} {_spell_number(_roman_to_int(m.group(2)))}",
        text,
    )


def _expand_symbols(text: str) -> str:
    text = text.replace("&", " dan ")
    # Slash between word-ish tokens -> "atau" (online/offline -> online atau offline).
    text = re.sub(r"(?<=\w)\s*/\s*(?=\w)", " atau ", text)
    text = text.replace("@", " at ")
    return text


def _spell_acronyms(text: str) -> str:
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        if token in _ACRONYM_WORD_WHITELIST:
            return token.lower()
        return _spell_letters(token)

    # Two or more consecutive uppercase letters.
    return re.sub(r"\b[A-Z]{2,}\b", _replace, text)


def normalize_for_tts(text: str) -> str:
    """Normalize raw model text into clean spoken Indonesian for TTS.

    Best-effort: on any failure the original text is returned so the user still
    hears something rather than silence.
    """
    if not text or not text.strip():
        return ""
    try:
        text = unicodedata.normalize("NFKC", text)
        # Remove zero-width / control chars (keep \n \t for markdown line rules).
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )
        for entity, word in _HTML_ENTITIES.items():
            text = text.replace(entity, word)
        text = _strip_markdown(text)
        text = _normalize_punctuation(text)
        text = _expand_emails_and_urls(text)
        text = _expand_roman(text)
        text = _expand_numbers(text)
        text = _spell_acronyms(text)
        text = _expand_symbols(text)
        text = text.lower()
        # Whitespace + spacing around punctuation cleanup.
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return text.strip()
    except Exception:
        logger.exception("TTS normalization failed; using raw text")
        return text.strip()
