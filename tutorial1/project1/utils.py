import re
import unicodedata

# Common non-ASCII punctuation mapped to an ASCII equivalent instead of being deleted
# (deleting them glued text together, e.g. "he’s" -> "hes", "2010–2015" -> "20102015").
_PUNCT_MAP = str.maketrans({
    "’": "'", "‘": "'", "‚": "'", "′": "'", "`": "'",
    "”": '"', "“": '"', "„": '"', "″": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...", "•": " ", "·": " ", "→": " ", "×": "x", "ʺ": '"',
    # letters that have no combining accent, so NFKD cannot strip them
    "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "þ": "th", "ð": "d",
})


def clean_text(text):
    # Normalize Unicode characters (e.g. full-width letters, ligatures)
    text = unicodedata.normalize("NFKC", text)

    # Map typographic punctuation to ASCII
    text = text.translate(_PUNCT_MAP)

    # Strip accents instead of dropping the letter: "Yaoundé" -> "Yaounde" (not "Yaound")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Whitespace control characters (\n, \t, \r) become a space, so words are not glued together
    text = re.sub(r"[\n\t\r\f\v]", " ", text)

    # Remove the remaining control / format characters (e.g. zero-width space, soft hyphen)
    text = "".join(
        c for c in text
        if unicodedata.category(c)[0] != "C"
    )

    # Keep ASCII only (what is left: symbols such as ®, ©, €, emoji, non-Latin scripts)
    text = re.sub(r"[^\x00-\x7F]", " ", text)

    # Collapse the whitespace we may have introduced
    return re.sub(r" {2,}", " ", text).strip()


def clean_corpus(corpus):
    return [clean_text(text) for text in corpus]
