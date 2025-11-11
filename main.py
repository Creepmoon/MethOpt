from __future__ import annotations

import sys
from pathlib import Path

from Sub.parser import parse_lp_file
from Sub.solver import solve_linear_program


def main() -> None:

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:

        input_path = Path("Sub") / "example4.txt"
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    try:
        spec = parse_lp_file(str(input_path))
        result = solve_linear_program(spec)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(2)

    output_lines = []
    if result.status == "optimal":
        output_lines.append("status: optimal")
        output_lines.append(f"objective: {result.objective_value:.10g}")
        output_lines.append("x*: " + " ".join(f"{v:.10g}" for v in result.variable_values or []))
    else:
        output_lines.append(f"status: {result.status}")
        if result.message:
            output_lines.append(result.message)
    out_text = "\n".join(output_lines)
    print(out_text)

    out_path = input_path.with_suffix(".out.txt")
    out_path.write_text(out_text, encoding="utf-8")


if __name__ == "__main__":
    main()


