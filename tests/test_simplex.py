# -*- coding: utf-8 -*-
"""simplex LP 求解器验证：手工案例、网格穷举交叉、scipy 交叉（如可用）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from cohortrag.simplex import lp_max

def close(a, b, tol=1e-7):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def test_basic_ub():
    st, x, o = lp_max(np.array([1.0, 1.0]),
                      np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
                      np.array([1.0, 0.6, 0.4]))
    assert st == "optimal" and close(o, 1.0) and close(x[0] + x[1], 1.0), (st, o, x)

def test_lower_bound_via_negative_b():
    st, x, o = lp_max(np.array([1.0]), np.array([[-1.0], [1.0]]), np.array([-5.0, 7.0]))
    assert st == "optimal" and close(o, 7.0), (st, o, x)

def test_eq_constraint():
    st, x, o = lp_max(np.array([1.0, 1.0]), None, None,
                      np.array([[1.0, 1.0]]), np.array([1.0]))
    assert st == "optimal" and close(o, 1.0) and close(x[0] + x[1], 1.0), (st, o, x)

def test_eq_negative_rhs():
    st, x, o = lp_max(np.array([1.0, 1.0]), None, None,
                      np.array([[-1.0, -1.0]]), np.array([-1.0]))
    assert st == "optimal" and close(o, 1.0), (st, o, x)

def test_infeasible():
    st, _, _ = lp_max(np.array([1.0]), np.array([[1.0], [-1.0]]), np.array([1.0, -2.0]))
    assert st == "infeasible", st

def test_unbounded():
    st, _, o = lp_max(np.array([1.0]))
    assert st == "unbounded" and np.isinf(o), (st, o)

def test_random_vs_grid():
    rng = np.random.default_rng(7)
    g = np.linspace(0, 1, 201)
    for _ in range(30):
        c = rng.uniform(-1, 1, 2)
        A = rng.uniform(-1, 1, (3, 2))
        A = np.vstack([A, [[1.0, 0.0], [0.0, 1.0]]])   # x,y <= 1，保证最优点在网格内
        x0 = rng.uniform(0.1, 0.5, 2)
        b = np.concatenate([A[:3] @ x0 + rng.uniform(0.2, 0.8, 3), [1.0, 1.0]])
        st, x, o = lp_max(c, A, b)
        assert st == "optimal", st
        assert np.all(A @ x <= b + 1e-6), "LP 解不可行"
        best = -1e18
        for i in g:
            for j in g:
                p = np.array([i, j])
                if np.all(A @ p <= b + 1e-9):
                    best = max(best, float(c @ p))
        assert o >= best - 0.06, (o, best)
        assert o <= best + 0.06, (o, best)

def test_scipy_crosscheck():
    try:
        from scipy.optimize import linprog
    except Exception:
        print("  (scipy 不可用，跳过该交叉验证)")
        return
    rng = np.random.default_rng(11)
    for _ in range(80):
        n = int(rng.integers(2, 6))
        m = int(rng.integers(1, 8))
        c = rng.uniform(-2, 2, n)
        A = rng.uniform(-2, 2, (m, n))
        b = rng.uniform(0.5, 2.0, m)
        st, x, o = lp_max(c, A, b)
        res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
        if st == "optimal":
            assert res.success, "scipy 认为不可行但 simplex 可行"
            assert abs(o - (-res.fun)) < 1e-6, (o, -res.fun)
        else:
            assert not res.success, f"simplex 说 {st} 但 scipy 可行"

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"共 {len(tests)} 项全部通过")
