"""
Guardrail actions — keyword/regex-based checks, tanpa extra LLM call.

Dua kategori:
  1. Jailbreak / persona override
  2. Konten berbahaya / tidak pantas
"""
import re

from nemoguardrails.actions import action

# ------------------------------------------------------------------ #
# Pattern lists                                                        #
# ------------------------------------------------------------------ #

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
    r"\bdan\b.*mode",
    r"developer mode",
    r"\bjailbreak\b",
    r"do anything now",
    r"(you are|you're) now (free|unrestricted|unfiltered|unchained)",
    # Indonesian — instruksi override
    r"lupakan (semua )?(instruksi|perintah)",
    r"abaikan (instruksi|perintah|sistem|aturan)",
    r"kamu (sekarang|adalah) (bebas|ai lain|robot baru|tanpa aturan|tanpa batasan)",
    r"tidak ada batasan",
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
    r"(cara|how to) (mencuri|merampok|menipu|memeras|blackmail|extort)",
    r"(cara|how to) (launder|mencuci) (money|uang)",
    r"(cara|how to) (scam|fraud|phishing) (orang|user|korban|victim)",
    r"(instruksi|tutorial) (untuk|for) (scam|penipuan|fraud|phishing)",
    r"(cara|how to) (berjudi|gambling) (illegal|ilegal)",
    # Penyalahgunaan zat
    r"(cara|how to) (membuat|meracik|synthesize) (narkoba|drugs|meth|heroin|cocaine|fentanyl|sabu)",
]


def _matches(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


# ------------------------------------------------------------------ #
# Actions                                                              #
# ------------------------------------------------------------------ #

@action(is_system_action=True)
async def is_jailbreak_attempt(context: dict) -> bool:
    return _matches(context.get("user_message", ""), _JAILBREAK)


@action(is_system_action=True)
async def is_harmful_content(context: dict) -> bool:
    return _matches(context.get("user_message", ""), _HARMFUL)
