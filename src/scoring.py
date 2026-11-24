"""Exact-match scoring and the R1/R3 rewards (proposal Section A.6)."""
import re

from el_core import ExecError, canon, execute, parse_program

LAMBDA = 0.5  # penalty for a wrong edit, both variants
TAU = 0.0     # abstention reward (R3 only). Without the Brier term abstaining wins when p < (tau+lam)/(1+lam) = 1/3;
              # with a calibrated Brier term the break-even is p = 0.5
BETA = 1.0    # Brier weight (R3 only)


def score(text, ex):
    """Return dict(valid, abstain, z, conf)."""
    conf = None
    c = re.search(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", text)
    if c:
        conf = min(1.0, max(0.0, float(c.group(1))))
    if re.search(r"<abstain\s*/>", text) and "<program>" not in text:
        return {"valid": True, "abstain": True, "z": 0, "conf": conf}
    try:
        prog, _ = parse_program(text)
        out = execute(prog, ex["before"])
    except (ExecError, Exception):
        return {"valid": False, "abstain": False, "z": 0, "conf": conf}
    z = int(canon(out) == canon(ex["after"]))
    return {"valid": True, "abstain": False, "z": z, "conf": conf}


def reward(text, ex, variant):
    s = score(text, ex)
    if not s["valid"]:
        return -1.0
    if variant == "R1":
        if s["abstain"]:
            return -1.0  # abstention is not an action in R1
        return s["z"] - LAMBDA * (1 - s["z"])
    # R3
    if s["abstain"]:
        return TAU
    c = 0.5 if s["conf"] is None else s["conf"]
    return s["z"] - LAMBDA * (1 - s["z"]) - BETA * (c - s["z"]) ** 2
