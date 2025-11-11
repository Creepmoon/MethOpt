import math
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class OptimizationResult:
	xt: float
	ft: float
	iterations: int
	elapsed_seconds: float
	samples: List[Tuple[float, float]]
	intervals: List[Tuple[float, float]]


def build_function_from_string(expr: str) -> Callable[[np.ndarray], np.ndarray]:

	expr = expr.strip()

	lower = expr.lower().replace(" ", "")
	if lower.startswith("f(x)="):
		expr = expr[expr.find("=") + 1:].strip()
	elif lower.startswith("y="):
		expr = expr[expr.find("=") + 1:].strip()


	allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

	allowed.update({
		"np": np,
		"sin": np.sin,
		"cos": np.cos,
		"tan": np.tan,
		"arcsin": np.arcsin,
		"arccos": np.arccos,
		"arctan": np.arctan,
		"exp": np.exp,
		"log": np.log,
		"log10": np.log10,
		"sqrt": np.sqrt,
		"abs": np.abs,
		"pi": math.pi,
		"e": math.e,
	})

	def f(x: np.ndarray) -> np.ndarray:
		local_vars = {"x": x}
		return eval(expr, {"__builtins__": {}}, {**allowed, **local_vars})

	return f


def estimate_lipschitz_constant(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, samples: int = 2048) -> float:
	x = np.linspace(a, b, samples, dtype=float)
	y = f(x)

	dx = np.diff(x)
	dy = np.diff(y)
	with np.errstate(divide="ignore", invalid="ignore"):
		slopes = np.abs(dy / dx)
	L = float(np.nanmax(slopes))

	if not np.isfinite(L) or L <= 0:
		L = 1.0
	return L


def piyavskii_shubert_minimize(
	f: Callable[[np.ndarray], np.ndarray],
	a: float,
	b: float,
	eps: float,
	L: Optional[float] = None,
	max_iters: int = 10000,
) -> OptimizationResult:
	if a >= b:
		raise ValueError("Левая граница должна быть меньше правой (a < b).")
	if eps <= 0:
		raise ValueError("Точность eps должна быть положительной.")

	if L is None:
		L = estimate_lipschitz_constant(f, a, b)

		L *= 1.05

	start_time = time.time()


	xs = [a, b]
	fs = [float(f(np.array([a]))[0]), float(f(np.array([b]))[0])]


	intervals: List[Tuple[float, float]] = [(a, b)]

	def lb_intersection(xl: float, fl: float, xr: float, fr: float) -> float:
		return 0.5 * (xr + xl) + 0.5 * (fr - fl) / L

	def characteristic(xl: float, fl: float, xr: float, fr: float) -> float:

		xm = lb_intersection(xl, fl, xr, fr)
		return 0.5 * (fl + fr - L * (xr - xl))

	iterations = 0
	while iterations < max_iters:
		iterations += 1

		order = np.argsort(xs)
		xs = [xs[i] for i in order]
		fs = [fs[i] for i in order]


		best_R = math.inf
		best_idx = None
		for i in range(len(xs) - 1):
			xl, xr = xs[i], xs[i + 1]
			fl, fr = fs[i], fs[i + 1]
			R = characteristic(xl, fl, xr, fr)
			if R < best_R:
				best_R = R
				best_idx = i

		assert best_idx is not None
		xl, xr = xs[best_idx], xs[best_idx + 1]
		fl, fr = fs[best_idx], fs[best_idx + 1]


		x_new = float(lb_intersection(xl, fl, xr, fr))

		x_new = max(a, min(b, x_new))


		if (xr - xl) <= eps:
			break

		f_new = float(f(np.array([x_new]))[0])

		xs.append(x_new)
		fs.append(f_new)

	# Результат
	best_idx = int(np.argmin(fs))
	xt, ft = xs[best_idx], fs[best_idx]
	elapsed = time.time() - start_time

	# Собираем историю и интервалы для визуализации
	samples = list(zip(xs, fs))
	intervals = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
	return OptimizationResult(xt=xt, ft=ft, iterations=iterations, elapsed_seconds=elapsed, samples=samples, intervals=intervals)


def plot_results(
	f: Callable[[np.ndarray], np.ndarray],
	a: float,
	b: float,
	result: OptimizationResult,
	L: float,
	title: str = "Глобальная минимизация (Пиявский–Шуберт)",
	show: bool = True,
	save_path: Optional[str] = None,
):
	x_grid = np.linspace(a, b, 2000)
	y_grid = f(x_grid)

	# Нижняя мажоранта (ломаная) как максимум опорных "V"-функций
	if result.samples:
		y_lb = np.full_like(x_grid, -np.inf, dtype=float)
		xs = np.array([p[0] for p in result.samples], dtype=float)
		fs = np.array([p[1] for p in result.samples], dtype=float)
		for xi, fi in zip(xs, fs):
			y_lb = np.maximum(y_lb, fi - L * np.abs(x_grid - xi))
	else:
		y_lb = None

	plt.figure(figsize=(10, 6))
	plt.plot(x_grid, y_grid, label="f(x)", color="#1f77b4", linewidth=2)

	if y_lb is not None and np.all(np.isfinite(y_lb)):
		plt.plot(x_grid, y_lb, label="Нижняя оценка (ломаная)", color="#ff7f0e", linewidth=1.5, linestyle="--")

	# Точки проб
	if result.samples:
		xs = [p[0] for p in result.samples]
		fs_ = [p[1] for p in result.samples]
		plt.scatter(xs, fs_, label="Пробные точки", color="#2ca02c", s=30, zorder=3)

	# Найденный минимум
	plt.scatter([result.xt], [result.ft], color="red", s=80, marker="*", label="Найденный минимум", zorder=5)
	plt.axvline(result.xt, color="red", linestyle=":", alpha=0.6)

	plt.title(title)
	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.grid(True, alpha=0.2)
	plt.legend()

	text_box = (
		f"x* ≈ {result.xt:.6g}\n"
		f"f(x*) ≈ {result.ft:.6g}\n"
		f"итераций: {result.iterations}\n"
		f"время: {result.elapsed_seconds:.3f} c\n"
		f"L ≈ {L:.6g}"
	)
	plt.gcf().text(0.015, 0.98, text_box, fontsize=10, va="top", ha="left",
	               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches="tight")
	if show:
		plt.show()
	else:
		plt.close()


def run_cli():
	print("Поиск глобального минимума (метод Пиявского–Шуберта)")
	print("Введите выражение функции (например, f(x) = x + sin(3.14159*x))")
	expr = input("f(x) = ").strip()
	if not expr:
		print("Строка пустая. Пример: x + sin(3.14159*x)")
		return
	try:
		a = float(input("Левая граница a: ").strip())
		b = float(input("Правая граница b: ").strip())
		eps = float(input("Точность eps (например 0.01): ").strip())
		L_input = input("Константа Липшица L (пусто — оценить автоматически): ").strip()
		L = float(L_input) if L_input else None
	except Exception as ex:
		print(f"Ошибка чтения параметров: {ex}")
		return

	f = build_function_from_string(expr)
	L_used = L if L is not None else estimate_lipschitz_constant(f, a, b) * 1.05
	result = piyavskii_shubert_minimize(f, a, b, eps, L=L_used)

	print("\nРезультаты:")
	print(f"  x* ≈ {result.xt}")
	print(f"  f(x*) ≈ {result.ft}")
	print(f"  итераций: {result.iterations}")
	print(f"  время: {result.elapsed_seconds:.3f} c")
	print(f"  использованный L ≈ {L_used}")

	title = "Глобальная минимизация: " + expr
	plot_results(f, a, b, result, L=L_used, title=title, show=True)


def demo_rastrigin():

	expr = "x**2 - 10*cos(2*pi*x) + 10"
	a, b = -5.12, 5.12
	eps = 0.01
	f = build_function_from_string(expr)
	L = estimate_lipschitz_constant(f, a, b) * 1.05
	result = piyavskii_shubert_minimize(f, a, b, eps, L=L)

	print("Демонстрация на функции Растригина (1D)")
	print(f"Функция: {expr}")
	print(f"[a, b] = [{a}, {b}], eps = {eps}, L ≈ {L:.4f}")
	print(f"x* ≈ {result.xt}, f(x*) ≈ {result.ft}")
	print(f"итераций: {result.iterations}, время: {result.elapsed_seconds:.3f} c")

	plot_results(f, a, b, result, L=L, title="Растригин (1D) — глобальная минимизация", show=True)


if __name__ == "__main__":
	try:
		run_cli()
	except EOFError:

		demo_rastrigin()


