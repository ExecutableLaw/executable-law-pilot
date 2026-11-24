import json

SYSTEM = """You apply U.S. regulatory amendments exactly. You are given an amendatory instruction from a Federal Register final rule (with any text the rule publishes) and the current text of one CFR section, split into numbered paragraphs. Write an edit program that turns the current section into the amended section.

The program is a JSON list of operations. Every index refers to the CURRENT (before) numbering:
  {"op": "replace", "p": i, "old": "...", "new": "..."}   replace the first occurrence of old in paragraph i with new
  {"op": "set", "p": i, "text": "..."}                     replace the whole of paragraph i
  {"op": "delete", "p": i}                                 remove paragraph i
  {"op": "insert_after", "p": i, "text": ["...", "..."]}   insert new paragraphs after paragraph i
Paragraph [0] is the section heading. "* * * * *" in published text means unchanged text is omitted.

Answer in exactly this format and nothing else:
<program>[...]</program>
<confidence>p</confidence>
where p in [0,1] is your probability that the program produces exactly the official amended text."""

SYSTEM_ABSTAIN = SYSTEM + """
If you think you are unlikely to be correct you may instead answer <abstain/> followed by <confidence>p</confidence>."""


def user_msg(ex):
    sec = "\n".join(f"[{i}] {p}" for i, p in enumerate(ex["before"]))
    return f"{ex['instruction']}\n\nCurrent text of § {ex['section']}:\n{sec}"


def target(ex, conf=0.5):
    return f"<program>{json.dumps(ex['ref_program'], ensure_ascii=False)}</program>\n<confidence>{conf}</confidence>"


def messages(ex, shots=(), abstain=False):
    m = [{"role": "system", "content": SYSTEM_ABSTAIN if abstain else SYSTEM}]
    for s in shots:
        m += [{"role": "user", "content": user_msg(s)}, {"role": "assistant", "content": target(s)}]
    m.append({"role": "user", "content": user_msg(ex)})
    return m
