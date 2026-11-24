"""Build the execution-verified amendment dataset for a set of CFR titles.

Usage: python build_data.py --titles 47 21 --out /data/out [--max-docs N]
Writes examples.jsonl (one per section-amendment that aligns) and funnel.json (alignment statistics).
"""
import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import threading
import time

import requests
from lxml import etree

sys.path.insert(0, os.path.dirname(__file__))
from el_core import (attested, canon, execute, norm, parse_ecfr_section, parse_fr_rule,
                     reference_program, render_instruction)

FR = "https://www.federalregister.gov/api/v1/documents.json"
ECFR = "https://www.ecfr.gov/api/versioner/v1"
S = requests.Session()
S.headers.update({"User-Agent": "execlaw-pilot/0.1 (research)", "Accept-Encoding": "gzip"})
_lock = threading.Lock()


MIN_INTERVAL = float(os.environ.get("MIN_INTERVAL", "0.35"))  # global politeness limit (seconds between requests)
_rl_lock = threading.Lock()
_last = [0.0]
STATS = collections.Counter()


def _wait_turn():
    with _rl_lock:
        now = time.time()
        wait = _last[0] + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def get(url, params=None, tries=8):
    for k in range(tries):
        _wait_turn()
        try:
            r = S.get(url, params=params, timeout=90)
            STATS[r.status_code] += 1
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429, 500, 502, 503, 504):
                with _rl_lock:  # back off globally
                    time.sleep(min(120, 15 * (k + 1)))
                continue
            return None
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    return None


def list_rules(title, d0, d1):
    out = []
    # FR API caps result windows, so walk year by year
    for y in range(int(d0[:4]), int(d1[:4]) + 1):
        a, b = max(d0, f"{y}-01-01"), min(d1, f"{y}-12-31")
        params = {"conditions[type][]": "RULE", "conditions[cfr][title]": title,
                  "conditions[effective_date][gte]": a, "conditions[effective_date][lte]": b,
                  "per_page": 1000, "fields[]": ["document_number", "effective_on", "full_text_xml_url",
                                                  "publication_date", "title", "agencies"]}
        url = FR
        while url:
            r = get(url, params)
            if r is None:
                break
            j = r.json()
            out += j.get("results", [])
            url, params = j.get("next_page_url"), None
    return out


_versions = {}


def part_versions(title, part):
    key = (title, part)
    with _lock:
        if key in _versions:
            return _versions[key]
    r = get(f"{ECFR}/versions/title-{title}.json", {"part": part})
    res = collections.defaultdict(list)
    if r is not None:
        for v in r.json().get("content_versions", []):
            if v.get("type") == "section":
                res[v["identifier"]].append(v)
    with _lock:
        _versions[key] = res
    return res


_parts = collections.OrderedDict()


def part_tree(title, part, date):
    key = (title, part, date)
    with _lock:
        if key in _parts:
            _parts.move_to_end(key)
            return _parts[key]
    r = get(f"{ECFR}/full/{date}/title-{title}.xml", {"part": part})
    tree = None
    if r is not None:
        try:
            tree = etree.fromstring(r.content)
        except Exception:
            tree = None
    with _lock:
        _parts[key] = tree
        while len(_parts) > 40:
            _parts.popitem(last=False)
    return tree


def section_xml(title, part, section, date):
    """Fetch the whole part once per date (cached) and cut the section out of it."""
    tree = part_tree(title, part, date)
    if tree is None:
        return None
    for el in tree.iter("DIV8"):
        if el.get("N") == section:
            return etree.tostring(el)
    return None


def process_doc(title, doc):
    """Return list of (status, record) for every section this rule amends."""
    r = get(doc["full_text_xml_url"]) if doc.get("full_text_xml_url") else None
    if r is None:
        return [("fr_xml_missing", {"doc": doc["document_number"]})]
    per = parse_fr_rule(r.content, title)
    d = doc.get("effective_on")
    out = []
    for sec, fr in per.items():
        base = {"doc": doc["document_number"], "section": sec, "date": d}
        if not fr["amdpars"]:
            out.append(("no_instruction", base)); continue
        if not d:
            out.append(("no_effective_date", base)); continue
        vers = part_versions(title, fr["part"]).get(sec, [])
        hit = [v for v in vers if v["date"] == d]
        if not hit:
            out.append(("no_ecfr_version_on_effective_date", base)); continue
        if hit[0].get("removed"):
            out.append(("section_removed", base)); continue
        prev = [v for v in vers if v["date"] < d]
        if not prev:
            out.append(("new_section", base)); continue
        before_date = (dt.date.fromisoformat(d) - dt.timedelta(days=1)).isoformat()
        bx, ax = section_xml(title, fr["part"], sec, before_date), section_xml(title, fr["part"], sec, d)
        if bx is None or ax is None:
            out.append(("ecfr_fetch_failed", base)); continue
        before, bf = parse_ecfr_section(bx)
        after, af = parse_ecfr_section(ax)
        if before is None or after is None:
            out.append(("ecfr_parse_failed", base)); continue
        rec = dict(base, title=title, part=fr["part"], amdpars=fr["amdpars"], supplied=fr["supplied"],
                   heading=fr.get("heading"), before=before, after=after,
                   instruction=render_instruction(fr), agencies=[a.get("name") for a in doc.get("agencies") or []],
                   flags={"table": bool(bf.get("table_or_math") or af.get("table_or_math") or fr.get("has_table")),
                          "other_tags": sorted(set(bf.get("other_tags", [])) | set(af.get("other_tags", [])))})
        if canon(before) == canon(after):
            out.append(("no_text_change", rec)); continue
        ok, why = attested(before, after, fr)
        if not ok:
            out.append((why, rec)); continue
        prog = reference_program(before, after)
        try:
            ok2 = canon(execute(prog, before)) == canon(after)
        except Exception:
            ok2 = False
        if not ok2:
            out.append(("reference_program_failed", rec)); continue
        rec["ref_program"] = prog
        out.append(("aligned", rec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2017-01-20")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    funnel = collections.Counter()
    by_title = collections.defaultdict(collections.Counter)
    fx = open(os.path.join(a.out, "examples.jsonl"), "w")
    fr_ = open(os.path.join(a.out, "rejects.jsonl"), "w")
    t0 = time.time()
    for title in a.titles:
        docs = list_rules(title, a.start, a.end)
        if a.max_docs:
            docs = docs[:a.max_docs]
        print(f"title {title}: {len(docs)} rules", flush=True)
        n = 0
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for res in ex.map(lambda d: process_doc(title, d), docs):
                n += 1
                for status, rec in res:
                    funnel[status] += 1
                    by_title[title][status] += 1
                    if status == "aligned":
                        fx.write(json.dumps(rec) + "\n")
                    else:
                        fr_.write(json.dumps({"status": status, **{k: rec.get(k) for k in ("doc", "section", "date", "amdpars", "supplied", "before", "after", "flags")}}) + "\n")
                if n % 20 == 0:
                    print(f"  {n}/{len(docs)} docs, {dict(by_title[title])}, http {dict(STATS)}, {time.time()-t0:.0f}s", flush=True)
                    fx.flush(); fr_.flush()
        json.dump({"total": funnel, "by_title": by_title}, open(os.path.join(a.out, "funnel.json"), "w"), indent=1)
    fx.close(); fr_.close()
    print("DONE", dict(funnel), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
