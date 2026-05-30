"""Deterministic Indonesian text normalization for the TTS frontend.

Turns noisy LLM output (markdown, symbols, raw numbers, gelar akademik,
singkatan) into clean spoken Indonesian with zero token cost.

Pipeline order:
    unicode → control-char → html → emoji → markdown → punctuation →
    gelar pre-nominal → gelar post-nominal → singkatan → email/url →
    waktu → ordinal → roman → angka → akronim → simbol →
    lowercase → whitespace
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
except Exception:  # pragma: no cover
    logger.warning("num2words unavailable; numbers will not be expanded for TTS")
    _HAS_NUM2WORDS = False

    def _spell_number(value: int) -> str:
        return str(value)


# Indonesian letter pronunciation for spelling out acronyms (PTN -> pe te en).
_LETTER_SOUNDS = {
    "A": "a",  "B": "be", "C": "ce",  "D": "de",  "E": "e",   "F": "ef",
    "G": "ge", "H": "ha", "I": "i",   "J": "je",  "K": "ka",  "L": "el",
    "M": "em", "N": "en", "O": "o",   "P": "pe",  "Q": "ki",  "R": "er",
    "S": "es", "T": "te", "U": "u",   "V": "fe",  "W": "we",  "X": "eks",
    "Y": "ye", "Z": "zet",
}

# All-caps tokens read as ordinary words, not spelled letter-by-letter.
_ACRONYM_WORD_WHITELIST = {
    "UDINUS", "UNDIP", "UNNES", "UGM", "UI", "ITB", "ITS", "UNS", "UB",
    "WIB", "WITA", "WIT",
    "BPJS", "KTP", "SIM", "STNK", "NPWP", "NIK", "NIM", "NIP", "NISN",
    "TNI", "POLRI", "KPU", "KPK", "DPR", "MPR",
    "SNMPTN", "SBMPTN", "PPDB", "UTBK",
    "COVID", "HIV", "AIDS",
    "PLN", "BUMN", "UMKM", "BRI", "BNI", "BTN", "BCA",
}

# Roman numerals only converted after context keywords.
# Keywords use (?i:...) for case-insensitive matching; the Roman numeral slot
# is case-sensitive (no flag) so lowercase words like "ini" or "iii" don't
# accidentally match.  (?=[IVXLCDM]) prevents zero-length matches because all
# sub-expressions allow 0 repetitions.
_ROMAN_KEYWORDS = r"bab|semester|jilid|kelas|abad|seri|tingkat|level|pasal|ayat"
_ROMAN_RE = re.compile(
    rf"\b((?i:{_ROMAN_KEYWORDS}))\s+"
    rf"((?=[IVXLCDM])M{{0,3}}(?:CM|CD|D?C{{0,3}})(?:XC|XL|L?X{{0,3}})(?:IX|IV|V?I{{0,3}}))\b",
)

_HTML_ENTITIES = {
    "&amp;": " dan ",
    "&lt;": " kurang dari ",
    "&gt;": " lebih dari ",
    "&quot;": '"',
    "&#39;": "'",
    "&nbsp;": " ",
}

# Emoji and pictograph stripping.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric extended
    "\U0001F800-\U0001F8FF"   # supplemental arrows
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess / medical
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\U00002600-\U000027BF"   # misc symbols (sun, star, etc.)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    re.UNICODE,
)

# ── Gelar / title tables ─────────────────────────────────────────────────────
# Longest / most specific compound forms MUST come before their components.
# Case-sensitive: Dr. (doktor) vs dr. (dokter umum); Drg. vs drg.

_PRE_NOMINAL_TITLES: list[tuple[str, str]] = [
    # Compound academic prefixes — order: longest first
    (r"\bProf\.\s*Dr\.\s*Ir\.",      "profesor doktor insinyur"),
    (r"\bProf\.\s*Dr\.\s*Drs\.",     "profesor doktor doktorandus"),
    (r"\bProf\.\s*Dr\.\s*Dra\.",     "profesor doktor doktoranda"),
    (r"\bProf\.\s*Dr\.",             "profesor doktor"),
    (r"\bProf\.\s*Ir\.",             "profesor insinyur"),
    (r"\bProf\.",                    "profesor"),
    # Medical / dental — lowercase dr/drg before uppercase Dr/Drg
    (r"\bdrg\.",                     "dokter gigi"),
    (r"\bDrg\.",                     "dokter gigi"),
    (r"\bdr\.",                      "dokter"),
    (r"\bDrs\.",                     "doktorandus"),
    (r"\bDra\.",                     "doktoranda"),
    (r"\bDr\.",                      "doktor"),
    (r"\bIr\.",                      "insinyur"),
    # Religious / honorific
    (r"\bKH\.",                      "kyai haji"),
    (r"\bHj\.\s+(?=[A-Z])",         "hajjah "),
    (r"\bH\.\s+(?=[A-Z])",          "haji "),   # only before capital (name)
    (r"\bUst\.",                     "ustadz"),
    (r"\bUstadzah\b",                "ustadzah"),
    (r"\bUstadz\b",                  "ustadz"),
    (r"\bKiai\b",                    "kiai"),
    # Nobility / Javanese
    (r"\bR\.A\.\s+(?=[A-Z])",       "raden ajeng "),
    (r"\bR\.M\.\s+(?=[A-Z])",       "raden mas "),
    (r"\bRr\.\s+(?=[A-Z])",         "rara "),
    (r"\bR\.\s+(?=[A-Z])",          "raden "),
    # Salutation
    (r"\bBpk\.",                     "bapak"),
    (r"\bSdri\.",                    "saudari"),
    (r"\bSdr\.",                     "saudara"),
    (r"\bNy\.",                      "nyonya"),
    (r"\bTn\.",                      "tuan"),
    (r"\bNn\.",                      "nona"),
    # Military / police rank (common ones)
    (r"\bMayjen\b",                  "mayor jenderal"),
    (r"\bLetjen\b",                  "letnan jenderal"),
    (r"\bBrigjen\b",                 "brigadir jenderal"),
    (r"\bIrjen\b",                   "inspektur jenderal"),
    (r"\bKomjen\b",                  "komisaris jenderal"),
    (r"\bKompol\b",                  "komisaris polisi"),
    (r"\bAKBP\b",                    "ajun komisaris besar polisi"),
    (r"\bKapolri\b",                 "kepala kepolisian republik indonesia"),
    (r"\bGen\.",                     "jenderal"),
    (r"\bKol\.",                     "kolonel"),
    (r"\bMay\.",                     "mayor"),
    (r"\bLet\.",                     "letnan"),
    (r"\bSerda\b",                   "sersan dua"),
    (r"\bSertu\b",                   "sersan satu"),
    (r"\bSerka\b",                   "sersan kepala"),
    (r"\bPraka\b",                   "prajurit kepala"),
]

_POST_NOMINAL_DEGREES: list[tuple[str, str]] = [
    # D3 / Ahli Madya — longer variants first
    (r"\bA\.Md\.T\.",               "ahli madya teknik"),
    (r"\bA\.Md\.Kom\.",             "ahli madya komputer"),
    (r"\bA\.Md\.Keb\.",             "ahli madya kebidanan"),
    (r"\bA\.Md\.Kep\.",             "ahli madya keperawatan"),
    (r"\bA\.Md\.Gz\.",              "ahli madya gizi"),
    (r"\bA\.Md\.",                  "ahli madya"),
    # S1 Sarjana — multi-component variants first
    (r"\bS\.Pd\.I\.",               "sarjana pendidikan islam"),
    (r"\bS\.KM\.",                  "sarjana kesehatan masyarakat"),
    (r"\bS\.Farm\.",                "sarjana farmasi"),
    (r"\bS\.Psi\.",                 "sarjana psikologi"),
    (r"\bS\.Sos\.",                 "sarjana sosial"),
    (r"\bS\.Ked\.",                 "sarjana kedokteran"),
    (r"\bS\.Kep\.",                 "sarjana keperawatan"),
    (r"\bS\.Kom\.",                 "sarjana komputer"),
    (r"\bS\.Pd\.",                  "sarjana pendidikan"),
    (r"\bS\.Gz\.",                  "sarjana gizi"),
    (r"\bS\.IP\.",                  "sarjana ilmu pemerintahan"),
    (r"\bS\.IK\.",                  "sarjana ilmu komunikasi"),
    (r"\bS\.Ak\.",                  "sarjana akuntansi"),
    (r"\bS\.Si\.",                  "sarjana sains"),
    (r"\bS\.Sn\.",                  "sarjana seni"),
    (r"\bS\.Ag\.",                  "sarjana agama"),
    (r"\bS\.Th\.",                  "sarjana teologi"),
    (r"\bS\.H\.",                   "sarjana hukum"),
    (r"\bS\.E\.",                   "sarjana ekonomi"),
    (r"\bS\.P\.",                   "sarjana pertanian"),
    (r"\bS\.T\.",                   "sarjana teknik"),
    # S2 Magister
    (r"\bM\.B\.A\.",                "master of business administration"),
    (r"\bM\.Eng\.",                 "master of engineering"),
    (r"\bM\.Sc\.",                  "master of science"),
    (r"\bM\.Pd\.I\.",               "magister pendidikan islam"),
    (r"\bM\.Kes\.",                 "magister kesehatan"),
    (r"\bM\.Hum\.",                 "magister humaniora"),
    (r"\bM\.Kom\.",                 "magister komputer"),
    (r"\bM\.Pd\.",                  "magister pendidikan"),
    (r"\bM\.Ak\.",                  "magister akuntansi"),
    (r"\bM\.Sn\.",                  "magister seni"),
    (r"\bM\.Si\.",                  "magister sains"),
    (r"\bM\.M\.",                   "magister manajemen"),
    (r"\bM\.H\.",                   "magister hukum"),
    (r"\bM\.E\.",                   "magister ekonomi"),
    (r"\bM\.T\.",                   "magister teknik"),
    (r"\bM\.Ag\.",                  "magister agama"),
    # S3 Doktor
    (r"\bPh\.D\.",                  "doktor"),
    (r"\bD\.Sc\.",                  "doktor sains"),
    # Profesi / spesialis kedokteran — multi-char codes first
    (r"\bSp\.OG\b",                 "spesialis obstetri ginekologi"),
    (r"\bSp\.PD\b",                 "spesialis penyakit dalam"),
    (r"\bSp\.JP\b",                 "spesialis jantung dan pembuluh darah"),
    (r"\bSp\.THT\b",                "spesialis telinga hidung tenggorok"),
    (r"\bSp\.KK\b",                 "spesialis kulit dan kelamin"),
    (r"\bSp\.Rad\b",                "spesialis radiologi"),
    (r"\bSp\.An\b",                 "spesialis anestesi"),
    (r"\bSp\.GK\b",                 "spesialis gizi klinik"),
    (r"\bSp\.KJ\b",                 "spesialis kedokteran jiwa"),
    (r"\bSp\.EM\b",                 "spesialis emergensi medik"),
    (r"\bSp\.N\b",                  "spesialis neurologi"),
    (r"\bSp\.M\b",                  "spesialis mata"),
    (r"\bSp\.U\b",                  "spesialis urologi"),
    (r"\bSp\.S\b",                  "spesialis saraf"),
    (r"\bSp\.A\b",                  "spesialis anak"),
    (r"\bSp\.B\b",                  "spesialis bedah"),
    # Profesi lainnya
    (r"\bApt\.",                    "apoteker"),
    (r"\bSKM\b",                    "sarjana kesehatan masyarakat"),
]

# ── Singkatan umum ────────────────────────────────────────────────────────────
# Multi-part abbreviations first, then single tokens.
_ABBREVIATIONS: list[tuple[str, str]] = [
    # Formal multi-part
    (r"\ba\.n\.",                   "atas nama"),
    (r"\bu\.p\.",                   "untuk perhatian"),
    (r"\bs\.d\.",                   "sampai dengan"),
    (r"\bs/d\b",                    "sampai dengan"),
    (r"\bd\.a\.",                   "dengan alamat"),
    (r"\by\.l\.",                   "yang lalu"),
    (r"\by\.a\.d\.",                "yang akan datang"),
    (r"\bttd\.",                    "tertanda"),
    (r"\bTTD\.",                    "tertanda"),
    # Common formal abbreviations
    (r"\bdll\.",                    "dan lain-lain"),
    (r"\bDLL\.",                    "dan lain-lain"),
    (r"\bdsb\.",                    "dan sebagainya"),
    (r"\bDSB\.",                    "dan sebagainya"),
    (r"\bdkk\.",                    "dan kawan-kawan"),
    (r"\bDKK\.",                    "dan kawan-kawan"),
    (r"\bybs\.",                    "yang bersangkutan"),
    (r"\btsb\.",                    "tersebut"),
    (r"\bTSB\.",                    "tersebut"),
    (r"\bsvp\.",                    "silakan"),
    (r"\bhal\.",                    "halaman"),
    (r"\bHal\.",                    "halaman"),
    (r"\bhlm\.",                    "halaman"),
    (r"\bNo\.\s*(?=\d)",            "nomor "),
    (r"\bno\.\s*(?=\d)",            "nomor "),
    (r"\bNomor\b",                  "nomor"),
    (r"\btgl\.",                    "tanggal"),
    (r"\bTgl\.",                    "tanggal"),
    # Address / location
    (r"\bJl\.",                     "jalan"),
    (r"\bjl\.",                     "jalan"),
    (r"\bGg\.",                     "gang"),
    (r"\bKab\.",                    "kabupaten"),
    (r"\bKec\.",                    "kecamatan"),
    (r"\bKel\.",                    "kelurahan"),
    (r"\bProv\.",                   "provinsi"),
    (r"\bGd\.",                     "gedung"),
    (r"\bLt\.\s*(?=\d)",            "lantai "),
    (r"\bRt\b",                     "rukun tetangga"),
    (r"\bRw\b",                     "rukun warga"),
    # Time / schedule
    (r"\bpkl\.",                    "pukul"),
    (r"\bpk\.",                     "pukul"),
    # Informal / chat abbreviations (word boundary)
    (r"\byg\b",                     "yang"),
    (r"\bdgn\b",                    "dengan"),
    (r"\butk\b",                    "untuk"),
    (r"\bkrn\b",                    "karena"),
    (r"\bshg\b",                    "sehingga"),
    (r"\bsdh\b",                    "sudah"),
    (r"\bblm\b",                    "belum"),
    (r"\bdpt\b",                    "dapat"),
    (r"\bspt\b",                    "seperti"),
    (r"\bsprti\b",                  "seperti"),
    (r"\bttg\b",                    "tentang"),
    (r"\bthd\b",                    "terhadap"),
    (r"\bthdp\b",                   "terhadap"),
    (r"\bkpd\b",                    "kepada"),
    (r"\bdri\b",                    "dari"),
    (r"\bjg\b",                     "juga"),
    (r"\bjgn\b",                    "jangan"),
    (r"\bbs\b",                     "bisa"),
    (r"\bkrg\b",                    "kurang"),
    (r"\bbrp\b",                    "berapa"),
    (r"\bsmua\b",                   "semua"),
    (r"\bbgt\b",                    "banget"),
    (r"\bbngt\b",                   "banget"),
    (r"\bsgt\b",                    "sangat"),
    (r"\bskrg\b",                   "sekarang"),
    (r"\bskrng\b",                  "sekarang"),
    (r"\bmnrt\b",                   "menurut"),
    (r"\bklo\b",                    "kalau"),
    (r"\bklau\b",                   "kalau"),
    (r"\bgimana\b",                 "bagaimana"),
    (r"\bkmrn\b",                   "kemarin"),
    (r"\bbsr\b",                    "besar"),
    (r"\bkcl\b",                    "kecil"),
    (r"\bsmpe\b",                   "sampai"),
    (r"\bsmpa\b",                   "sampai"),
    (r"\bbtl\b",                    "betul"),
    (r"\bmsk\b",                    "masuk"),
]

# Indonesian month name abbreviations → full name.
_MONTH_ABBR: dict[str, str] = {
    "Jan": "januari", "Feb": "februari", "Mar": "maret",
    "Apr": "april",   "Mei": "mei",      "Jun": "juni",
    "Jul": "juli",    "Agt": "agustus",  "Agust": "agustus",
    "Sep": "september", "Okt": "oktober", "Nov": "november",
    "Des": "desember",
}
_MONTH_ABBR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _MONTH_ABBR) + r")\b"
)


# ── Helper functions ──────────────────────────────────────────────────────────

def _roman_to_int(roman: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(roman.upper()):
        cur = values.get(ch, 0)
        total += -cur if cur < prev else cur
        prev = max(prev, cur)
    return total


def _spell_letters(token: str) -> str:
    return " ".join(_LETTER_SOUNDS.get(ch, ch) for ch in token.upper())


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9]*\n?", " ", text)
    text = text.replace("`", "")
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}\d+\.\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}([-*_])\1{2,}\s*$", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    return text


def _normalize_punctuation(text: str) -> str:
    for src, dst in {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "–": "-", "…": ".",
    }.items():
        text = text.replace(src, dst)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{3,}", ".", text)
    return text


def _expand_gelar(text: str) -> str:
    """Expand academic/honorific titles (pre- and post-nominal)."""
    for pattern, replacement in _PRE_NOMINAL_TITLES:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in _POST_NOMINAL_DEGREES:
        text = re.sub(pattern, replacement, text)
    return text


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    # Month abbreviations
    text = _MONTH_ABBR_RE.sub(lambda m: _MONTH_ABBR[m.group(1)], text)
    return text


def _expand_emails_and_urls(text: str) -> str:
    def _email(m: re.Match) -> str:
        local, domain = m.group(1), m.group(2)
        return f"{local} at {domain.replace('.', ' dot ')}"

    text = re.sub(r"\b([\w.+-]+)@([\w.-]+\.\w+)\b", _email, text)

    def _url(m: re.Match) -> str:
        return m.group(0).replace(".", " dot ")

    # Require alphabetic TLD so "25.000" (thousands) is not treated as domain.
    text = re.sub(r"\b[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b", _url, text)
    return text


def _expand_time(text: str) -> str:
    """Expand time expressions to spoken form.

    Handles:
      - pukul 08.00 / pukul 08:00 WIB  -> "pukul delapan wib"
      - jam 14:30                       -> "jam empat belas tiga puluh"  (no double jam)
      - 08:30                           -> "jam delapan tiga puluh"
      - 08.30 (dot separator, context)  -> "jam delapan tiga puluh"
    """
    def _render(hour: int, minute: int) -> str:
        h = _spell_number(hour)
        return h if minute == 0 else f"{h} {_spell_number(minute)}"

    # "pukul HH.MM" or "pukul HH:MM" with optional timezone
    def _pukul(m: re.Match) -> str:
        spoken = _render(int(m.group(1)), int(m.group(2)))
        tz = (" " + m.group(3).lower()) if m.group(3) else ""
        return f"pukul {spoken}{tz}"

    text = re.sub(
        r"\bpukul\s+([01]?\d|2[0-3])[.:]([0-5]\d)(?:\s+(WIB|WITA|WIT))?\b",
        _pukul, text, flags=re.IGNORECASE,
    )

    # "jam HH:MM" — expand digits but keep "jam" from the input (no double)
    def _jam_colon(m: re.Match) -> str:
        return f"jam {_render(int(m.group(1)), int(m.group(2)))}"

    text = re.sub(
        r"\bjam\s+([01]?\d|2[0-3]):([0-5]\d)\b",
        _jam_colon, text,
    )

    # Standalone HH:MM (colon is unambiguous — add "jam" prefix)
    def _colon_time(m: re.Match) -> str:
        return f"jam {_render(int(m.group(1)), int(m.group(2)))}"

    text = re.sub(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
        _colon_time, text,
    )

    # HH.MM with dot: only expand when clearly time (not inside a larger number).
    def _dot_time(m: re.Match) -> str:
        return f"jam {_render(int(m.group(1)), int(m.group(2)))}"

    text = re.sub(
        r"(?<!\d)\b([01]?\d|2[0-3])\.([0-5]\d)\b(?!\d)",
        _dot_time, text,
    )
    return text


def _expand_ordinal(text: str) -> str:
    """ke-1 -> pertama, ke-2 -> kedua, ke-N -> ke<N-in-words>."""
    _SPECIAL = {1: "pertama", 2: "kedua", 3: "ketiga", 4: "keempat",
                5: "kelima", 6: "keenam", 7: "ketujuh", 8: "kedelapan",
                9: "kesembilan", 10: "kesepuluh", 11: "kesebelas",
                12: "kedua belas"}

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if n in _SPECIAL:
            return _SPECIAL[n]
        return f"ke{_spell_number(n)}"

    return re.sub(r"\bke-(\d+)\b", _replace, text)


def _expand_roman(text: str) -> str:
    return _ROMAN_RE.sub(
        lambda m: f"{m.group(1)} {_spell_number(_roman_to_int(m.group(2)))}",
        text,
    )


def _expand_numbers(text: str) -> str:
    # Currency: Rp 25.000 -> "dua puluh lima ribu rupiah".
    def _currency(m: re.Match) -> str:
        digits = m.group(1).replace(".", "").replace(",", "")
        try:
            return f"{_spell_number(int(digits))} rupiah"
        except ValueError:
            return m.group(0)

    text = re.sub(r"\bRp\.?\s?(\d[\d.,]*)", _currency, text, flags=re.IGNORECASE)

    # Percent: 10% / 10,5% -> "sepuluh persen".
    text = re.sub(
        r"(\d[\d.,]*)\s*%",
        lambda m: f"{_expand_number_token(m.group(1))} persen",
        text,
    )

    # Dimension / multiplication: 4x6 / 1920×1080 -> "... kali ...".
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

    # Remaining standalone numbers.
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
    # Integer with optional dot thousands separators.
    cleaned = token.replace(".", "")
    if not cleaned.isdigit():
        return token
    try:
        return _spell_number(int(cleaned))
    except ValueError:
        return token


def _expand_symbols(text: str) -> str:
    text = text.replace("&", " dan ")
    text = re.sub(r"(?<=\w)\s*/\s*(?=\w)", " atau ", text)
    text = text.replace("@", " at ")
    return text


def _spell_acronyms(text: str) -> str:
    def _replace(m: re.Match) -> str:
        token = m.group(0)
        if token in _ACRONYM_WORD_WHITELIST:
            return token.lower()
        return _spell_letters(token)

    return re.sub(r"\b[A-Z]{2,}\b", _replace, text)


def _normalize_repeated_chars(text: str) -> str:
    """Collapse informal repeated lowercase chars: 'baikkk' -> 'baikk'.

    Only matches lowercase ASCII to avoid collapsing uppercase Roman numerals
    (III, VIII) or other legitimate all-caps sequences before they are expanded.
    """
    return re.sub(r"([a-z])\1{2,}", r"\1\1", text)


# ── Public entry point ────────────────────────────────────────────────────────

def normalize_for_tts(text: str) -> str:
    """Normalize raw model/user text into clean spoken Indonesian for TTS.

    Best-effort: on any failure the original text is returned so the user
    still hears something rather than silence.
    """
    if not text or not text.strip():
        return ""
    try:
        # 1. Unicode canonicalization
        text = unicodedata.normalize("NFKC", text)
        # 2. Strip zero-width / control chars (keep \n \t for markdown rules).
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )
        # 3. HTML entities
        for entity, word in _HTML_ENTITIES.items():
            text = text.replace(entity, word)
        # 4. Emoji
        text = _strip_emoji(text)
        # 5. Markdown
        text = _strip_markdown(text)
        # 6. Punctuation normalization
        text = _normalize_punctuation(text)
        # 7. Informal repeated chars (baikkk -> baik)
        text = _normalize_repeated_chars(text)
        # 8. Academic / honorific titles — before number/acronym steps
        text = _expand_gelar(text)
        # 9. Common abbreviations
        text = _expand_abbreviations(text)
        # 10. Email / URL
        text = _expand_emails_and_urls(text)
        # 11. Time expressions (before number expansion to protect HH.MM)
        text = _expand_time(text)
        # 12. Ordinal ke-N
        text = _expand_ordinal(text)
        # 13. Roman numerals (contextual)
        text = _expand_roman(text)
        # 14. Numbers
        text = _expand_numbers(text)
        # 15. Acronyms
        text = _spell_acronyms(text)
        # 16. Remaining symbols
        text = _expand_symbols(text)
        # 17. Lowercase
        text = text.lower()
        # 18. Whitespace cleanup
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return text.strip()
    except Exception:
        logger.exception("TTS normalization failed; using raw text")
        return text.strip()
