from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set


@dataclass
class LinearProgramSpec:
    sense: str
    coefficients_by_var: Dict[str, float]
    constraints: List[Tuple[Dict[str, float], str, float]]
    free_variables: Set[str]


_token_re = re.compile(r"([+\-]?\s*\d*\.?\d*(?:e[+\-]?\d+)?)\s*\*?\s*([A-Za-z]\w*)")


def _parse_linear_expr(expr: str) -> Dict[str, float]:
    cleaned = expr.replace("−", "-")

    if not cleaned.strip():
        return {}

    cleaned = cleaned.replace("-", "+-")
    terms = [t.strip() for t in cleaned.split("+") if t.strip()]
    result: Dict[str, float] = {}
    for term in terms:

        m = _token_re.fullmatch(term.replace(" ", ""))
        if not m:

            if re.fullmatch(r"-?[A-Za-z]\w*", term):
                coef = -1.0 if term.strip().startswith("-") else 1.0
                var = term.strip().lstrip("+-").strip()
                result[var] = result.get(var, 0.0) + coef
                continue

            if re.fullmatch(r"-?\d*\.?\d+(?:e[+\-]?\d+)?", term, re.IGNORECASE):
                continue
            raise ValueError(f"Cannot parse linear term: '{term}' inside '{expr}'")
        coef_text, var = m.groups()
        coef = float(coef_text) if coef_text not in ("", "+", "-") else (1.0 if coef_text in ("", "+") else -1.0)
        result[var] = result.get(var, 0.0) + coef
    # Удаляем нулевые коэффициенты после возможной компенсации
    result = {k: v for k, v in result.items() if abs(v) > 1e-12}
    return result


def parse_lp_file(path: str) -> LinearProgramSpec:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    if not lines:
        raise ValueError("Input file is empty")
    obj_line = lines[0]
    m = re.match(r"(?i)\s*(min|max)\s*:?\s*(.+)$", obj_line)
    if not m:
        raise ValueError("Первая строка должна быть целью: 'min:' или 'max:'")
    sense = m.group(1).lower()
    obj_expr = m.group(2)
    obj_map = _parse_linear_expr(obj_expr)
    i = 1
    constraints: List[Tuple[Dict[str, float], str, float]] = []
    free_vars: Set[str] = set()

    def consume_constraint(line: str) -> bool:
        m2 = re.match(r"(.+?)(<=|>=|=)(.+)", line)
        if not m2:
            return False
        lhs_text, sign, rhs_text = m2.groups()
        lhs = _parse_linear_expr(lhs_text)
        rhs = float(rhs_text.replace(" ", "").replace(",", "."))
        constraints.append((lhs, sign, rhs))
        return True


    if i < len(lines) and re.match(r"(?i)^subject\s+to:?", lines[i]):
        i += 1


    while i < len(lines):
        line = lines[i]
        if re.match(r"(?i)^free\s*:", line):

            tail = line.split(":", 1)[1]
            names = [v.strip() for v in tail.split(",") if v.strip()]
            free_vars.update(names)
            i += 1
            break
        if not consume_constraint(line):
            raise ValueError(f"Cannot parse line {i+1}: '{line}'")
        i += 1


    while i < len(lines) and re.match(r"(?i)^free\s*:", lines[i]):
        tail = lines[i].split(":", 1)[1]
        names = [v.strip() for v in tail.split(",") if v.strip()]
        free_vars.update(names)
        i += 1

    return LinearProgramSpec(
        sense=sense,
        coefficients_by_var=obj_map,
        constraints=constraints,
        free_variables=free_vars,
    )


