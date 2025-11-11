from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import math

from .parser import LinearProgramSpec


@dataclass
class StandardForm:

    A: List[List[float]]
    b: List[float]
    c: List[float]

    var_names: List[str]
    original_var_names: List[str]
    original_to_standard: Dict[str, Tuple[int, int]]

    artificial_indices: Set[int]

    initial_basis: List[int]


def _clone_matrix(M: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in M]


def to_standard_form(spec: LinearProgramSpec) -> StandardForm:

    var_set: Set[str] = set(spec.coefficients_by_var.keys())
    for lhs, _, _ in spec.constraints:
        var_set.update(lhs.keys())
    original_vars: List[str] = sorted(var_set)


    var_names: List[str] = []
    original_to_standard: Dict[str, Tuple[int, int]] = {}
    for v in original_vars:
        if v in spec.free_variables:
            pos_idx = len(var_names)
            var_names.append(f"{v}_pos")
            neg_idx = len(var_names)
            var_names.append(f"{v}_neg")
            original_to_standard[v] = (pos_idx, neg_idx)
        else:
            idx = len(var_names)
            var_names.append(v)
            original_to_standard[v] = (idx, -1)


    A: List[List[float]] = []
    b: List[float] = []
    initial_basis: List[int] = []
    artificial_indices: Set[int] = set()

    def expand_coeff_map(coeffs: Dict[str, float]) -> List[float]:
        row = [0.0] * len(var_names)
        for orig, coef in coeffs.items():
            pos_idx, neg_idx = original_to_standard[orig]
            if neg_idx == -1:
                row[pos_idx] += coef
            else:

                row[pos_idx] += coef
                row[neg_idx] -= coef
        return row


    c_base = [0.0] * len(var_names)
    for orig, coef in spec.coefficients_by_var.items():
        pos_idx, neg_idx = original_to_standard[orig]
        if spec.sense.lower() == "min":
            coef = -coef
        if neg_idx == -1:
            c_base[pos_idx] += coef
        else:
            c_base[pos_idx] += coef
            c_base[neg_idx] -= coef


    slack_count = 0
    for lhs, sign, rhs in spec.constraints:
        row = expand_coeff_map(lhs)

        if rhs < 0:
            row = [-v for v in row]
            rhs = -rhs

            if sign == "<=":
                sign = ">="
            elif sign == ">=":
                sign = "<="
            else:
                sign = "="

        if sign == "<=":

            slack_idx = len(var_names)
            var_names.append(f"s{slack_count}")
            slack_count += 1

            for r in A:
                r.append(0.0)
            row.append(1.0)

            c_base.append(0.0)
            A.append(row)
            b.append(rhs)
            initial_basis.append(slack_idx)
        elif sign == ">=":

            surplus_idx = len(var_names)
            var_names.append(f"t{slack_count}")
            slack_count += 1
            for r in A:
                r.append(0.0)
            row.append(-1.0)
            c_base.append(0.0)

            art_idx = len(var_names)
            var_names.append(f"a{len(artificial_indices)}")
            for r in A:
                r.append(0.0)
            row.append(1.0)
            c_base.append(0.0)
            A.append(row)
            b.append(rhs)
            initial_basis.append(art_idx)
            artificial_indices.add(art_idx)
        elif sign == "=":

            art_idx = len(var_names)
            var_names.append(f"a{len(artificial_indices)}")
            for r in A:
                r.append(0.0)
            row.append(1.0)
            c_base.append(0.0)
            A.append(row)
            b.append(rhs)
            initial_basis.append(art_idx)
            artificial_indices.add(art_idx)
        else:
            raise ValueError(f"Unknown constraint sign: {sign}")

    return StandardForm(
        A=A,
        b=b,
        c=c_base,
        var_names=var_names,
        original_var_names=original_vars,
        original_to_standard=original_to_standard,
        artificial_indices=artificial_indices,
        initial_basis=initial_basis,
    )


@dataclass
class SimplexResult:
    status: str
    objective_value: Optional[float]
    variable_values: Optional[List[float]]
    message: str = ""


class TwoPhaseSimplex:
    def __init__(self, sf: StandardForm) -> None:
        self.sf = sf

    def _build_phase1(self) -> Tuple[List[List[float]], List[float], List[int], List[float]]:
        A = _clone_matrix(self.sf.A)
        b = self.sf.b[:]
        basis = self.sf.initial_basis[:]
        num_vars = len(self.sf.var_names)


        c_phase1 = [0.0] * num_vars
        for j in self.sf.artificial_indices:
            c_phase1[j] = -1.0

        return A, b, basis, c_phase1

    @staticmethod
    def _choose_entering(reduced_costs: List[float], epsilon: float = 1e-9) -> Optional[int]:
        best_j = None
        best_val = 0.0
        for j, rc in enumerate(reduced_costs):
            if rc > epsilon and (best_j is None or rc > best_val + 1e-12):
                best_j = j
                best_val = rc
        return best_j

    @staticmethod
    def _ratio_test(column: List[float], b: List[float], epsilon: float = 1e-9) -> Tuple[Optional[int], float]:
        min_ratio = math.inf
        pivot_row = None
        for i, aij in enumerate(column):
            if aij > epsilon:
                ratio = b[i] / aij
                if ratio < min_ratio - 1e-12:
                    min_ratio = ratio
                    pivot_row = i
        return pivot_row, min_ratio

    def _compute_reduced_costs(self, A: List[List[float]], b: List[float], basis: List[int], costs: List[float]) -> Tuple[List[float], List[float]]:

        m = len(basis)
        n = len(costs)


        B = [[A[i][basis[k]] for k in range(m)] for i in range(m)]


        c_B = [costs[basis[i]] for i in range(m)]
        y = _solve_linear_system_transposed(B, c_B)

        reduced_costs = [costs[j] - sum(y[i] * A[i][j] for i in range(m)) for j in range(n)]
        obj_value = sum(y[i] * b[i] for i in range(m))
        return reduced_costs, [obj_value]

    def _pivot(self, A: List[List[float]], b: List[float], basis: List[int], pivot_row: int, pivot_col: int) -> None:
        m = len(b)
        n = len(A[0])
        piv = A[pivot_row][pivot_col]
        inv = 1.0 / piv
        A[pivot_row] = [v * inv for v in A[pivot_row]]
        b[pivot_row] *= inv
        for i in range(m):
            if i == pivot_row:
                continue
            factor = A[i][pivot_col]
            if abs(factor) < 1e-12:
                continue
            A[i] = [A[i][j] - factor * A[pivot_row][j] for j in range(n)]
            b[i] -= factor * b[pivot_row]
        basis[pivot_row] = pivot_col

    def _simplex(self, A: List[List[float]], b: List[float], basis: List[int], costs: List[float]) -> Tuple[str, float, List[float]]:
        m = len(b)
        n = len(A[0])
        while True:
            reduced_costs, obj_val_wrapper = self._compute_reduced_costs(A, b, basis, costs)
            entering = self._choose_entering(reduced_costs)
            if entering is None:
                x = [0.0] * n
                for i, bi in enumerate(basis):
                    x[bi] = b[i]
                return "optimal", obj_val_wrapper[0], x
            column = [A[i][entering] for i in range(m)]
            pivot_row, _ = self._ratio_test(column, b)
            if pivot_row is None:
                return "unbounded", math.inf, []
            self._pivot(A, b, basis, pivot_row, entering)

    def solve(self) -> SimplexResult:

        A, b, basis, c_phase1 = self._build_phase1()
        status1, obj1, x1 = self._simplex(A, b, basis, c_phase1)
        if status1 == "unbounded":
            return SimplexResult(status="infeasible", objective_value=None, variable_values=None, message="Фаза I неограничена (вырожденный ввод).")

        sum_artificial = sum(x1[j] for j in self.sf.artificial_indices)
        if sum_artificial > 1e-7:
            return SimplexResult(status="infeasible", objective_value=None, variable_values=None, message="Допустимого решения нет: искусственные переменные > 0.")


        keep_indices = [j for j in range(len(self.sf.var_names)) if j not in self.sf.artificial_indices]

        new_index_of = {j: k for k, j in enumerate(keep_indices)}
        A2 = [[row[j] for j in keep_indices] for row in A]
        c2 = [self.sf.c[j] for j in keep_indices]
        basis2 = []
        for i, bi in enumerate(basis):
            if bi in self.sf.artificial_indices:

                swaped = False
                for j in range(len(keep_indices)):
                    if A2[i][j] > 1e-9 and j not in basis2:
                        basis2.append(j)

                        self._pivot(A2, b, basis2, i, j)
                        swaped = True
                        break
                if not swaped:

                    pass
            else:
                basis2.append(new_index_of[bi])


        status2, obj2, x2 = self._simplex(A2, b, basis2, c2)
        if status2 == "unbounded":
            return SimplexResult(status="unbounded", objective_value=None, variable_values=None, message="Целевая функция неограниченна.")


        obj_val = obj2
        if hasattr(self.sf, "c"):
            pass

        return SimplexResult(status="optimal", objective_value=obj_val, variable_values=x2)


def _solve_linear_system_transposed(A: List[List[float]], b: List[float]) -> List[float]:

    m = len(A)

    At = [[A[j][i] for j in range(m)] for i in range(m)]
    for i in range(m):
        At[i].append(b[i])


    for col in range(m):

        pivot_row = max(range(col, m), key=lambda r: abs(At[r][col]))
        if abs(At[pivot_row][col]) < 1e-12:
            raise ValueError("Сингулярная матрица при вычислении базиса")

        if pivot_row != col:
            At[col], At[pivot_row] = At[pivot_row], At[col]

        piv = At[col][col]
        inv = 1.0 / piv
        At[col] = [v * inv for v in At[col]]

        for r in range(m):
            if r == col:
                continue
            factor = At[r][col]
            if abs(factor) < 1e-12:
                continue
            At[r] = [At[r][k] - factor * At[col][k] for k in range(m + 1)]

    return [At[i][m] for i in range(m)]


def solve_linear_program(spec: LinearProgramSpec) -> SimplexResult:
    sf = to_standard_form(spec)
    solver = TwoPhaseSimplex(sf)
    res = solver.solve()

    if res.status == "optimal" and res.variable_values is not None:
        num_original_std = max(max(idx, (neg if neg != -1 else 0)) for idx, neg in sf.original_to_standard.values()) + 1
        std_values = res.variable_values[:num_original_std]
        original_values: Dict[str, float] = {}
        for orig, (pos_idx, neg_idx) in sf.original_to_standard.items():
            if neg_idx == -1:
                original_values[orig] = std_values[pos_idx] if pos_idx < len(std_values) else 0.0
            else:
                pos_val = std_values[pos_idx] if pos_idx < len(std_values) else 0.0
                neg_val = std_values[neg_idx] if neg_idx < len(std_values) else 0.0
                original_values[orig] = pos_val - neg_val

        obj = sum(spec.coefficients_by_var.get(var, 0.0) * original_values.get(var, 0.0) for var in sf.original_var_names)
        if spec.sense == "min":
            res.objective_value = obj
        else:
            res.objective_value = obj
        res.variable_values = [original_values[var] for var in sf.original_var_names]
    return res


