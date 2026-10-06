"""Input guard: refuse jailbreak attempts and plainly harmful requests.

Keyword and regex checks, no model call. These began as the patterns the NeMo
Guardrails flow ran. The flow added nothing on top of them — no LLM rail, no
dialogue rail — but its runtime cost ~5–10 ms on every turn (measured against
0.02 ms for the regexes alone), pulled in a heavy dependency, and answered in
the voice of a different bot ("Bengbot").

Several patterns were narrowed because they refused ordinary Indonesian: the
kiosk's visitors ask "apakah tidak ada batasan usia?" and "mode gelap dan mode
terang", and were told Luna could not change her identity. Each narrowing is
pinned in tests/unit/test_agent_guard.py, in both directions.

A refusal is returned as text for the agent to speak and to persist, so the
conversation history shows what was said rather than a gap.
"""

import re

_JAILBREAK = [
    # English — instruction override
    r"ignore (your |all |previous )?instructions",
    r"forget (your )?instructions",
    r"disregard (your )?(previous |all )?instructions",
    r"override (your )?(system |instructions|prompt)",
    r"bypass (your )?(restrictions|rules|guidelines|safety|filter)",
    r"(you are|you have) no (restrictions|rules|limitations|ethics)",
    r"without (any )?(restrictions|ethical|moral|limits|filter)",
    r"(show|reveal|print|display|repeat|output) (your |the )?(system prompt|instructions|prompt)",
    r"repeat (everything|all) (above|before|from the top)",
    r"ignore (safety|ethical) (guidelines|rules|constraints)",
    r"unlock (your |the )?restrictions",
    # Persona hijack
    r"pretend (you are|to be)",
    r"act as (dan|dve|jailbreak|unrestricted|evil|villain|free ai)",
    r"roleplay as (a |an )?(different|other|unrestricted|evil|hacker)",
    r"new persona",
    # Not "developer mode" alone: that is how anyone asks about their phone.
    r"(you are|you're|kamu) (now |sekarang )?(in |dalam )?(developer|dev) mode",
    r"\bjailbreak (you|yourself|kamu|dirimu|luna|the ai|this ai|model)\b",
    r"do anything now",
    r"(you are|you're) now (free|unrestricted|unfiltered|unchained)",
    # Indonesian — instruksi override
    r"lupakan (semua )?(instruksi|perintah)",
    r"abaikan (instruksi|perintah|sistem|aturan)",
    r"kamu (sekarang|adalah) (bebas|ai lain|robot baru|tanpa aturan|tanpa batasan)",
    # Addressed to the assistant. "Tidak ada batasan" on its own is how a
    # visitor asks whether there is an age limit to enrol.
    r"(kamu|anda|kau|dirimu) (sekarang )?(tidak (punya|ada|memiliki) batasan|tanpa (batasan|aturan))",
    r"(tampilkan|cetak|ungkap|print|output) (sistem prompt|instruksi sistem|promptmu)",
    r"ubah (identitas|karaktermu|dirimu) menjadi",
    r"berpura-pura (jadi|menjadi|kamu adalah)",
    r"pura-pura (jadi|menjadi|sebagai)",
]

_HARMFUL = [
    # Senjata & kekerasan
    r"(cara|how to|tutorial) (membuat|merakit|buat|synthesize|create|make) (bom|peledak|senjata api|granat|racun|bahan peledak|explosive|bomb|weapon|poison|methamphetamine|meth|drugs|narkoba)",
    r"(cara|how to) (membunuh|menyakiti|melukai|torture|menyiksa) (orang|seseorang|manusia|someone|people)",
    r"(cara|how to) (bunuh diri|self.?harm|menyiksa diri|overdose)",
    r"(instructions|guide|tutorial) (for|to) (hurt|harm|kill|attack|abuse|murder)",
    # Siber & hacking
    r"(cara|how to) (hack|membobol|meretas|crack) (sistem|account|password|server|database|website)",
    r"(cara|how to) (mencuri data|steal data|exfiltrate data)",
    r"(sql|code|script) injection (attack|exploit|tutorial)",
    r"(malware|ransomware|trojan|keylogger|rootkit|worm|virus) (code|script|sample|tutorial|create|buat|tulis)",
    r"(cara|how to) (bypass|disable) (antivirus|firewall|security|waf)",
    # Konten dewasa / CSAM
    r"(konten|materi|gambar|video|foto) (pornografi|dewasa|porno|hentai|seksual eksplisit)",
    r"(explicit|sexual|nude|naked) (content|material|image|video|photo)",
    r"(child|underage|minor|anak) (pornography|sexual|abuse|nude)",
    # Kriminal & penipuan
    # With an object. "Mencuri perhatian" is catching someone's eye, and
    # "memeras" is also to squeeze an orange.
    r"(cara|how to) (mencuri|merampok|membobol) (uang|motor|mobil|sepeda|barang|rumah|toko|bank|hp|ponsel|dompet|laptop|kendaraan)",
    r"(cara|how to) (menipu|memeras) (orang|seseorang|korban|teman|keluarga|someone|people)",
    r"(how to) (blackmail|extort)",
    r"(cara|how to) (launder|mencuci) (money|uang)",
    r"(cara|how to) (scam|fraud|phishing) (orang|user|korban|victim)",
    r"(instruksi|tutorial) (untuk|for) (scam|penipuan|fraud|phishing)",
    r"(cara|how to) (berjudi|gambling) (illegal|ilegal)",
    # Penyalahgunaan zat
    r"(cara|how to) (membuat|meracik|synthesize) (narkoba|drugs|meth|heroin|cocaine|fentanyl|sabu)",
]

_JAILBREAK_RE = [re.compile(p) for p in _JAILBREAK]
# Matched against the ORIGINAL text, not the lowercased one: "DAN" the jailbreak
# is written in capitals, and "dan" is the Indonesian for "and" — lowercased,
# "mode gelap dan mode terang" was refused as an attack.
_JAILBREAK_CASED_RE = [re.compile(r"\bDAN\b.*\b(?i:mode)\b")]
_HARMFUL_RE = [re.compile(p) for p in _HARMFUL]

REFUSE_JAILBREAK = (
    "Maaf, saya tidak bisa mengubah identitas atau mengabaikan instruksi saya. "
    "Saya siap membantu pertanyaan seputar Universitas Dian Nuswantoro atau topik umum lainnya!"
)
REFUSE_HARMFUL = (
    "Maaf, permintaan tersebut tidak dapat saya proses. Saya hanya bisa membantu "
    "dengan hal-hal yang aman dan bermanfaat. Silakan ajukan pertanyaan lain."
)


def is_jailbreak(text: str | None) -> bool:
    original = text or ""
    lowered = original.lower()
    return any(pattern.search(lowered) for pattern in _JAILBREAK_RE) or any(
        pattern.search(original) for pattern in _JAILBREAK_CASED_RE
    )


def is_harmful(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(pattern.search(lowered) for pattern in _HARMFUL_RE)


def refusal_for(text: str | None) -> str | None:
    """The sentence to answer with instead of the agent, or None to proceed."""
    if is_jailbreak(text):
        return REFUSE_JAILBREAK
    if is_harmful(text):
        return REFUSE_HARMFUL
    return None
