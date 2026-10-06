"""Split Qwen output into reasoning and final answer, then pull out code and exploit."""

import re
from dataclasses import dataclass

_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_EXPLOIT = re.compile(r"<exploit>\n?(.*?)</exploit>", re.DOTALL)


@dataclass
class Parsed:
    reasoning: str
    final: str
    code: str | None
    exploit: str | None
    truncated_reasoning: bool  # never closed </think>


def parse(text: str) -> Parsed:
    if "</think>" in text:
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
