"""Mechanical acronym audit of paper/AFDM_TVT.tex.

For every acronym (2+ capitals, optional digits/hyphens) appearing in the body
text, report it unless its first occurrence is the definition "(ACR)" or it is
whitelisted (units, standard symbols). Title and keywords are excluded
from 'first use' because IEEE re-defines acronyms in the abstract and body.
"""
import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent / "paper" / "AFDM_TVT.tex"
WHITE = {"IEEE", "II", "III", "IV", "V", "VI", "VII", "TX", "USA", "GHz", "MHz", "kHz", "SNR-independent"}


def body(s):
    s = re.sub(r"(?m)%.*$", "", s)                       # comments
    s = re.sub(r"\\begin\{(equation|align|algorithmic)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)                       # inline math
    s = re.sub(r"\\(label|ref|eqref|cite|input|url|bibliography\w*|includegraphics)\{[^}]*\}", " ", s)
    return s


def main():
    s = TEX.read_text()
    a = s.index(r"\begin{abstract}")
    kw = s.index(r"\end{IEEEkeywords}")
    abstract = body(s[a:s.index(r"\end{abstract}")])
    main_text = body(s[kw:])
    bad = 0
    for name, text in (("abstract", abstract), ("body", main_text)):
        seen = set()
        for m in re.finditer(r"\b([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*s?)\b", text):
            acr = m.group(1).rstrip("s") if m.group(1).endswith("s") and m.group(1)[:-1].isupper() else m.group(1)
            if "-" in acr and acr.split("-")[0] in seen:
                continue                                   # e.g. CRC-16 after CRC is defined
            if acr in WHITE or acr in seen or len(acr) < 2:
                continue
            seen.add(acr)
            pre = text[max(0, m.start() - 1): m.end() + 1]
            if not pre.startswith("(") and f"({acr})" not in text[max(0, m.start() - 200): m.start()]:
                ctx = text[max(0, m.start() - 60): m.end() + 20].replace("\n", " ")
                print(f"[{name}] {acr}: first use not a definition ...{ctx}...")
                bad += 1
    print(f"{bad} acronym problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
