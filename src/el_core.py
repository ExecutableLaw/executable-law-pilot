"""Executable Law pilot core: section parsing, edit-program language, interpreter, reference programs.

A section is a list of paragraphs (strings). Index 0 is the section heading.
An edit program is a JSON list of operations, every index refers to the BEFORE text:
  {"op": "replace", "p": i, "old": "...", "new": "..."}   first occurrence of old in paragraph i -> new
  {"op": "set", "p": i, "text": "..."}                     paragraph i becomes text
  {"op": "delete", "p": i}                                 paragraph i removed
  {"op": "insert_after", "p": i, "text": ["...", ...]}     new paragraphs after paragraph i (in order)
"""
import difflib
import json
import re
import unicodedata

from lxml import etree

# ---------------------------------------------------------------- text normalisation

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "--").replace("–", "-")
    s = _WS.sub(" ", s).strip()
    s = _PAREN_IN.sub("(", s)
    s = _PAREN_OUT.sub(")", s)
    return s


_PAREN_IN = re.compile(r"\(\s+")
_PAREN_OUT = re.compile(r"\s+\)")
_LABEL = re.compile(r"^(\([0-9a-zA-Z]{1,6}\)\s*)+")


def strip_label(p):
    return _LABEL.sub("", p).strip()


def canon(paras):
    return "\n".join(norm(p) for p in paras if norm(p))


def elem_text(e) -> str:
    return norm("".join(e.itertext()))


# ---------------------------------------------------------------- eCFR section XML

BAD_TAGS = {"GPOTABLE", "TABLE", "MATH", "GPH", "IMG", "img", "table", "FORMULA"}
PARA_TAGS = {"P", "FP", "NOTE", "EXTRACT"}


def parse_ecfr_section(xml_bytes):
    """Return (paragraphs, flags) for an eCFR versioner section XML. None if unusable."""
    try:
        root = etree.fromstring(xml_bytes)
    except Exception:
        return None, {"parse_error": True}
    sec = root if root.get("TYPE") == "SECTION" else root.find(".//DIV8")
    if sec is None:
        return None, {"no_section": True}
    flags = {}
    if next(sec.iter(*BAD_TAGS), None) is not None:
        flags["table_or_math"] = True
    paras = []
    for ch in sec:
        tag = ch.tag if isinstance(ch.tag, str) else ""
        if tag == "HEAD":
            paras.append(elem_text(ch))
        elif tag in ("CITA", "XREF", "EDNOTE", "SECAUTH", "APPRO", "EFFDNOT"):
            continue
        elif tag in PARA_TAGS or tag.startswith("P"):
            if tag in ("NOTE", "EXTRACT"):
                for sub in ch:
                    if isinstance(sub.tag, str) and sub.tag not in ("XREF",):
                        t = elem_text(sub)
                        if t:
                            paras.append(t)
            else:
                t = elem_text(ch)
                if t:
                    paras.append(t)
        elif tag and tag not in ("DIV9",):
            flags.setdefault("other_tags", set()).add(tag)
    if "other_tags" in flags:
        flags["other_tags"] = sorted(flags["other_tags"])
    if not paras:
        return None, {"empty": True}
    return paras, flags


# ---------------------------------------------------------------- Federal Register rule XML

SEC_RE = re.compile(r"(\d+[a-z]?\.\d+[a-z]?(?:-\d+)?)")


def sections_in_amdpar(text, part):
    """Section numbers mentioned in an amendatory instruction (only those in this part)."""
    t = norm(text)
    out = []
    # handle "§ 1.2", "§§ 1.2 and 1.3", "section 1.2", ranges "1.2 through 1.5" are not expanded
    for m in re.finditer(r"(§+|[Ss]ections?)\s*([^:;]*)", t):
        chunk = m.group(2)
        for s in SEC_RE.findall(chunk):
            if s.split(".")[0] == str(part) and s not in out:
                out.append(s)
    return out


def parse_fr_rule(xml_bytes, title):
    """Yield dicts {section, amdpars:[...], supplied:[paragraph strings or '* * *'], heading}
    grouped per section for REGTEXT blocks of this title."""
    try:
        root = etree.fromstring(xml_bytes)
    except Exception:
        return {}
    per = {}
    for rt in root.iter("REGTEXT"):
        if rt.get("TITLE") != str(title):
            continue
        part = rt.get("PART")
        cur = []  # sections referenced by the most recent AMDPAR
        for ch in rt:
            tag = ch.tag if isinstance(ch.tag, str) else ""
            if tag == "AMDPAR":
                txt = elem_text(ch)
                if re.search(r"authority", txt, re.I) and not re.search(r"§", txt):
                    cur = []
                    continue
                secs = sections_in_amdpar(txt, part)
                if secs:
                    cur = secs
                elif not cur:
                    continue
                # an instruction without a section number continues the previous one ("a. Revising paragraph (b)")
                for s in cur:
                    per.setdefault(s, {"section": s, "part": part, "amdpars": [], "supplied": [], "heading": None})
                    per[s]["amdpars"].append(txt)
            elif tag == "SECTION":
                sn = ch.find("SECTNO")
                if sn is None:
                    continue
                ids = SEC_RE.findall(elem_text(sn))
                if not ids:
                    continue
                s = ids[0]
                d = per.setdefault(s, {"section": s, "part": part, "amdpars": [], "supplied": [], "heading": None})
                subj = ch.find("SUBJECT")
                if subj is not None:
                    d["heading"] = norm(f"§ {s} {elem_text(subj)}")
                for sub in ch:
                    st = sub.tag if isinstance(sub.tag, str) else ""
                    if st in ("SECTNO", "SUBJECT"):
                        continue
                    if st == "STARS":
                        d["supplied"].append("* * * * *")
                    elif st in BAD_TAGS:
                        d["supplied"].append("[TABLE]")
                        d["has_table"] = True
                    elif st in ("EXTRACT", "NOTE"):
                        for x in sub:
                            if isinstance(x.tag, str):
                                t = elem_text(x)
                                if t:
                                    d["supplied"].append(t)
                    else:
                        t = elem_text(sub)
                        if t:
                            d["supplied"].append(t)
            elif tag in BAD_TAGS:
                for s in cur:
                    per[s]["has_table"] = True
    return per


# ---------------------------------------------------------------- interpreter


class ExecError(Exception):
    pass


def execute(program, before):
    """Apply an edit program to a list of paragraphs. Raises ExecError on any invalid operation."""
    if not isinstance(program, list):
        raise ExecError("program must be a list")
    n = len(before)
    cur = list(before)
    deleted = set()
    inserts = {}
    touched = set()
    for op in program:
        if not isinstance(op, dict) or "op" not in op:
            raise ExecError("bad op")
        k = op["op"]
        p = op.get("p")
        if not isinstance(p, int) or isinstance(p, bool) or not (0 <= p < n):
            raise ExecError(f"bad index {p}")
        if k == "replace":
            old, new = op.get("old"), op.get("new")
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                raise ExecError("replace needs old/new")
            if p in deleted:
                raise ExecError("replace on deleted")
            i = cur[p].find(old)
            if i < 0:
                raise ExecError("old text not found")
            cur[p] = cur[p][:i] + new + cur[p][i + len(old):]
        elif k == "set":
            t = op.get("text")
            if not isinstance(t, str):
                raise ExecError("set needs text")
            if p in deleted:
                raise ExecError("set on deleted")
            cur[p] = t
        elif k == "delete":
            if p in deleted:
                raise ExecError("double delete")
            deleted.add(p)
        elif k == "insert_after":
            t = op.get("text")
            if isinstance(t, str):
                t = [t]
            if not isinstance(t, list) or not all(isinstance(x, str) for x in t) or not t:
                raise ExecError("insert needs text list")
            inserts.setdefault(p, []).extend(t)
        else:
            raise ExecError(f"unknown op {k}")
        touched.add(p)
    out = []
    for i in range(n):
        if i not in deleted:
            out.append(cur[i])
        out.extend(inserts.get(i, []))
    return out


def parse_program(text):
    """Extract the JSON program (and optional confidence) from model output.
    Expected output: <program>[...]</program> optionally <confidence>0.x</confidence>."""
    m = re.search(r"<program>(.*?)</program>", text, re.S)
    if not m:
        raise ExecError("no <program> block")
    try:
        prog = json.loads(m.group(1).strip())
    except Exception as e:
        raise ExecError(f"json: {e}")
    conf = None
    c = re.search(r"<confidence>\s*([0-9.]+)\s*</confidence>", text)
    if c:
        try:
            conf = min(1.0, max(0.0, float(c.group(1))))
        except ValueError:
            conf = None
    return prog, conf


# ---------------------------------------------------------------- reference program from a diff


def _replace_op(i, a, b):
    """Smallest 'replace' that turns paragraph a into b, with old's first occurrence at the diff site.
    Returns None if a 'set' is the better encoding."""
    pre = 0
    while pre < min(len(a), len(b)) and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < min(len(a), len(b)) - pre and a[len(a) - 1 - suf] == b[len(b) - 1 - suf]:
        suf += 1
    s, e_a, e_b = pre, len(a) - suf, len(b) - suf
    # expand to word boundaries
    while s > 0 and a[s - 1] not in " ":
        s -= 1
    while e_a < len(a) and a[e_a] not in " ":
        e_a += 1
        e_b += 1
    # ensure old non-empty and its first occurrence is at s
    for _ in range(40):
        old = a[s:e_a]
        if old and a.find(old) == s:
            break
        # grow left by one word, else right
        if s > 0:
            s -= 1
            while s > 0 and a[s - 1] != " ":
                s -= 1
        elif e_a < len(a):
            e_a += 1
            e_b += 1
            while e_a < len(a) and a[e_a] != " ":
                e_a += 1
                e_b += 1
        else:
            break
    old, new = a[s:e_a], b[s:e_b]
    if not old or a.find(old) != s:
        return None
    if len(old) > 0.6 * len(a) and len(a) > 40:
        return None
    return {"op": "replace", "p": i, "old": old, "new": new}


def reference_program(before, after):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    prog = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            prog += [{"op": "delete", "p": i} for i in range(i1, i2)]
        elif tag == "insert":
            if i1 == 0:  # never insert before the heading; attach after heading
                prog.append({"op": "insert_after", "p": 0, "text": after[j1:j2]})
            else:
                prog.append({"op": "insert_after", "p": i1 - 1, "text": after[j1:j2]})
        else:  # replace
            na, nb = i2 - i1, j2 - j1
            k = min(na, nb)
            for t in range(k):
                i, a, b = i1 + t, before[i1 + t], after[j1 + t]
                op = _replace_op(i, a, b)
                prog.append(op if op else {"op": "set", "p": i, "text": b})
            if na > k:
                prog += [{"op": "delete", "p": i} for i in range(i1 + k, i2)]
            if nb > k:
                prog.append({"op": "insert_after", "p": i1 + k - 1, "text": after[j1 + k:j2]})
    return prog


# ---------------------------------------------------------------- attestation check


def _expand_words(text, run):
    """Grow a diff run to whole words of text; strip punctuation."""
    run = run.strip()
    if not run:
        return ""
    i = text.find(run)
    if i < 0:
        return run.strip(" ,;.")
    j = i + len(run)
    while i > 0 and text[i - 1] not in " ":
        i -= 1
    while j < len(text) and text[j] not in " ":
        j += 1
    return text[i:j].strip(" ,;.")


def attested(before, after, fr):
    """Is every change between before and after attested by the rule text for this section?
    Returns (ok, reason)."""
    supplied = {norm(x) for x in fr["supplied"] if x not in ("* * * * *", "[TABLE]")}
    if fr.get("heading"):
        supplied.add(norm(fr["heading"]))
    amd = norm(" ".join(fr["amdpars"]))
    amd_l = amd.lower()
    bset = set(before)
    redesig = "redesignat" in amd_l
    b_unlabeled = {strip_label(x) for x in before}
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("delete", "replace") and (i2 - i1) > (j2 - j1):
            if not re.search(r"remov|redesignat|revis|reserv|delet|amend", amd_l):
                return False, "unattested_delete"
        new_paras = after[j1:j2]
        olds = before[i1:i2]
        for t, b in enumerate(new_paras):
            if b in bset or b in supplied:
                continue
            if redesig and strip_label(b) in b_unlabeled and strip_label(b):
                continue
            if "reserv" in amd_l and re.fullmatch(r"(\([0-9a-zA-Z]{1,6}\)\s*-?\s*)+\[Reserved\]", b):
                continue
            # derivable from paired old paragraph by an attested substitution?
            if t < len(olds):
                a = olds[t]
                pre = 0
                while pre < min(len(a), len(b)) and a[pre] == b[pre]:
                    pre += 1
                suf = 0
                while suf < min(len(a), len(b)) - pre and a[len(a) - 1 - suf] == b[len(b) - 1 - suf]:
                    suf += 1
                # every inserted word-run must be quoted in the instruction as whole words (>= 3 chars)
                runs = [b[j1_:j2_] for tg, i1_, i2_, j1_, j2_ in
                        difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if tg in ("replace", "insert")]
                runs = [_expand_words(b, r) for r in runs]
                if runs and all(r == "" or (len(r) >= 3 and re.search(r"(?<!\w)" + re.escape(r) + r"(?!\w)", amd)) for r in runs):
                    continue
            return False, "unattested_text"
    return True, "ok"


def render_section(paras):
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(paras))


def render_instruction(fr):
    lines = ["Amendatory instruction(s):"]
    lines += [f"- {a}" for a in fr["amdpars"]]
    if fr["supplied"] or fr.get("heading"):
        lines.append("")
        lines.append("Text published with the instruction:")
        if fr.get("heading"):
            lines.append(fr["heading"])
        lines += fr["supplied"]
    return "\n".join(lines)
