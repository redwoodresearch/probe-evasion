"""Split Qwen output into reasoning and final answer, then pull out code and exploit."""

import re
from dataclasses import dataclass

_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_TAG, _END = "<exploit>", "</exploit>"

# One definition shared by the sandbox and the probes: the exploit section starts at the LAST <exploit> tag (the
# answer's prose can mention "the <exploit> tag" earlier, common with reasoning off), and the program is the last
# code block before it. So the code the probe reads is exactly the code the sandbox runs.


def strip_exploit(final: str) -> str:
    """The final answer up to the exploit section. Probes read only this span: the tag itself
    is a lexical giveaway, and (by causal attention) earlier tokens never see it."""
    return final.rsplit(_TAG, 1)[0].rstrip()


def program(final: str) -> str | None:
    blocks = _CODE.findall(strip_exploit(final))
    return blocks[-1] if blocks else None


def exploit_input(final: str) -> str | None:
    start = final.rfind(_TAG)
    end = final.find(_END, start) if start >= 0 else -1
    return _clean_exploit(final[start + len(_TAG):end]) if end >= 0 else None


def code_span(response: str) -> tuple[int, int] | None:
    """Char span of the last code block's body in `response`. Probes read only this span, so
    prose announcing a flaw ("this program contains a deliberate bug...") can't be detected
    lexically and dropping that prose doesn't count as evasion."""
    matches = list(_CODE.finditer(response))
    return matches[-1].span(1) if matches else None


@dataclass
class Parsed:
    reasoning: str
    final: str
    code: str | None
    exploit: str | None
    truncated_reasoning: bool  # never closed </think>


def parse(text: str, has_reasoning: bool = True) -> Parsed:
    """has_reasoning=False for models whose text output is the final answer (e.g. Claude)."""
    if not has_reasoning:
        reasoning, final, truncated = "", text, False
    elif "</think>" in text:
        reasoning, final = text.rsplit("</think>", 1)
        truncated = False
    else:
        reasoning, final, truncated = text, "", True
    reasoning = reasoning.removeprefix("<think>").strip()

    final = final.strip()
    return Parsed(reasoning, final, program(final), exploit_input(final), truncated)


def _clean_exploit(raw: str) -> str:
    lines = raw.strip("\n").split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip("\n") + "\n"


def neutralize(example: dict) -> dict:
    """The same code replayed under a bare "problem -> code block" context: no system prompt, no reasoning,
    no prose. Isolates what the code itself carries from what context (disclosures, plans) leaks into it."""
    code = example["response"][slice(*example["code_span"])]
    response = f"```python\n{code}```"
    out = {k: v for k, v in example.items() if k not in ("system", "reasoning", "response", "code_span")}
    return out | {"reasoning": "", "response": response, "code_span": code_span(response)}


def _without_docstrings(tree):
    import ast

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return tree


def strip_comments(code: str) -> str:
    """The same program with comments and docstrings removed (and lines left empty by that dropped). Asserts the
    syntax tree, ignoring docstrings, is unchanged, so only non-executable text differs."""
    import ast
    import io
    import tokenize

    lines = code.splitlines(keepends=True)
    cuts = []  # (start (row, col), end (row, col)) spans to blank
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT:
            cuts.append((tok.start, tok.end))
        elif tok.type == tokenize.STRING:
            prev = next((t for t in reversed(tokens[:i]) if t.type not in (tokenize.NL, tokenize.COMMENT)), None)
            nxt = next((t for t in tokens[i + 1:] if t.type not in (tokenize.NL, tokenize.COMMENT)), None)
            statement_start = prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            if statement_start and nxt is not None and nxt.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                cuts.append((tok.start, tok.end))  # a bare string statement: a docstring or a comment-like string
    chars = [list(l) for l in lines]
    for (r0, c0), (r1, c1) in cuts:
        for r in range(r0, r1 + 1):
            row = chars[r - 1]
            lo, hi = (c0 if r == r0 else 0), (c1 if r == r1 else len(row) - (row[-1:] == ["\n"]))
            for c in range(lo, hi):
                row[c] = " "
    out = []
    for before, after in zip(lines, ("".join(r) for r in chars)):
        if after.strip() or not before.strip():
            out.append(after.rstrip() + ("\n" if after.endswith("\n") else ""))
    stripped = "".join(out)
    # A docstring-only body becomes `pass`-less; compare against the original with docstrings removed the same way.
    assert ast.dump(_without_docstrings(ast.parse(code))) == ast.dump(_without_docstrings(ast.parse(stripped))), "code changed"
    return stripped

