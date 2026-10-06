"""Split Qwen output into reasoning and final answer, then pull out code and exploit."""

import re
from dataclasses import dataclass

_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_EXPLOIT = re.compile(r"<exploit>\n?(.*?)</exploit>", re.DOTALL)


def strip_exploit(final: str) -> str:
    """The final answer up to the exploit tag. Probes read only this span: the tag itself
    is a lexical giveaway, and (by causal attention) earlier tokens never see it."""
    return final.split("<exploit>", 1)[0].rstrip()


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

    # Drop exploit sections first so a fenced exploit input isn't mistaken for the program.
    codes = _CODE.findall(_EXPLOIT.sub("", final))
    code = codes[-1] if codes else None
    exploits = _EXPLOIT.findall(final)
    exploit = _clean_exploit(exploits[-1]) if exploits else None
    return Parsed(reasoning, final.strip(), code, exploit, truncated)


def _clean_exploit(raw: str) -> str:
    lines = raw.strip("\n").split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip("\n") + "\n"
