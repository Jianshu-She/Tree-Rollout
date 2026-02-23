import re
from typing import List


# ------------------------------------------------------------------
# Terminal detection
# ------------------------------------------------------------------

def _has_completed_boxed(text: str) -> bool:
    """Check if text contains a completed \\boxed{...} (with matching braces)."""
    start = 0
    while True:
        pos = text.find("\\boxed{", start)
        if pos == -1:
            return False
        brace = 1
        i = pos + 7
        while i < len(text) and brace > 0:
            if text[i] == "{":
                brace += 1
            elif text[i] == "}":
                brace -= 1
            i += 1
        if brace == 0:
            return True
        start = pos + 1


def is_terminal_text(text: str) -> bool:
    """Check if the *cumulative* reasoning contains a completed final answer.

    We require a fully closed \\boxed{...} so that we don't stop the tree
    before the actual answer content is generated.
    """
    return _has_completed_boxed(text)


# ------------------------------------------------------------------
# Answer extraction (ported from data-prepare/simple_math_evaluator.py)
# ------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...}, handling nested braces."""
    matches = []
    start = 0
    while True:
        start_pos = text.find("\\boxed{", start)
        if start_pos == -1:
            break
        brace_count = 1
        pos = start_pos + 7  # len('\\boxed{')
        content_start = pos
        while pos < len(text) and brace_count > 0:
            if text[pos] == "{":
                brace_count += 1
            elif text[pos] == "}":
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            matches.append(text[content_start : pos - 1].strip())
        start = start_pos + 1
    return matches[-1] if matches else ""


def extract_answer(response: str) -> str:
    """Extract the final answer from a model response."""
    # Try \\boxed{} first
    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # Fallback patterns
    patterns = [
        r"[Tt]herefore[,:]?\s*([^\n\.]+)",
        r"[Tt]hus[,:]?\s*([^\n\.]+)",
        r"[Ss]o[,:]?\s*([^\n\.]+)",
        r"[Aa]nswer[:]?\s*([^\n\.]+)",
        r"[Ff]inal[ly]?[,:]?\s*([^\n\.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()

    # Last resort: last line with math content
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if line and any(c in line for c in "=()[]{}^+-*/%0123456789"):
            return line
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = answer.strip().rstrip(".")
    answer = answer.replace("\\left(", "(").replace("\\right)", ")")
    answer = answer.replace("\\left[", "[").replace("\\right]", "]")
    answer = answer.replace("\\{", "{").replace("\\}", "}")
    answer = answer.replace("\\frac", "frac")
    answer = re.sub(r"\$+", "", answer)
    answer = answer.replace("\\\\", "")
    return answer.strip()


def is_correct(model_response: str, ground_truth: str) -> bool:
    """Check if model's extracted answer matches ground truth."""
    extracted = normalize_answer(extract_answer(model_response))
    truth = normalize_answer(ground_truth)
    if not extracted:
        return False
    if extracted == truth:
        return True
    # Whitespace-insensitive
    if extracted.replace(" ", "") == truth.replace(" ", ""):
        return True
    # Numeric comparison
    try:
        clean_e = re.sub(r"[^0-9.\-+]", "", extracted)
        clean_t = re.sub(r"[^0-9.\-+]", "", truth)
        if clean_e and clean_t:
            if abs(float(clean_e) - float(clean_t)) < 1e-6:
                return True
    except (ValueError, TypeError):
        pass
    return False
