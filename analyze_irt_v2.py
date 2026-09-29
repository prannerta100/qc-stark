"""2PL IRT for QC-Stark.

Model
-----
    P(Y_mjs = 1) = 1 / (1 + exp(-a_j * (theta_m - b_j)))

    theta_m   ability of model m,            m = 1..10   ->  10 params
    b_j       difficulty of item j,          j = 1..55    ->  55 params
    a_j       discrimination of item j                    ->  55 params

An item is one (task, level) cell. The 5 seeds are replicate administrations
of item j to model m, so they carry no parameter of their own: each (m, j)
contributes k successes out of n = 5 Bernoulli trials at the same p_mj.

    120 free parameters, 2750 observations, 50 observations per item.

Identification
--------------
The 2PL likelihood is unchanged by theta -> (theta-d)/c, b -> (b-d)/c,
a -> c*a. Scale is fixed by standardising theta to mean 0, SD 1 after every
sweep, with a and b rescaled to compensate.

Estimation
----------
Penalised joint maximum likelihood, alternating:
  * each theta_m by 1-D Newton over its 55 items
  * each (a_j, b_j) by 2-D Newton over its 10 models

Why penalised: with only 10 persons, 55 free discriminations are not
identifiable. Unpenalised JMLE does not converge on this data -- after 400
sweeps the largest parameter change is still 0.39, the log-likelihood cycles
between -992 and -1006, and 20 of the 55 discriminations sit pinned at the
search bounds. A log-normal prior on a_j, equivalent to adding

    -(LAMBDA/2) * (log a_j)^2

to the log-likelihood, regularises the slopes. LAMBDA = 4 corresponds to a
log-normal(0, 0.5) prior, the usual weakly-informative default, and converges
in well under 100 sweeps with no discrimination at a bound. Set --lam 0 to see
the unpenalised behaviour.

Standard errors come from the penalised observed information at the optimum.
They are conditional -- the ability SEs treat the item parameters as known and
vice versa -- so they understate joint uncertainty.

Usage:  python analyze_irt_v2.py [--latex]
"""
import json, math, argparse, collections
import numpy as np

RESULTS = "results/full_benchmark_final.jsonl"
TASK_LABEL = {"A_stateprep":"State Prep","G1_trotter":"Trotter","H1_oracle":"Oracle",
              "B1_debugging":"Debugging","B3_noise_logic":"Noise Discrim","J1_reverse":"Reverse Eng",
              "F1_equivalence":"Equivalence","C1_routing":"Routing","I1_noise":"Noise Fid",
              "E2_vqe":"VQE","D1_qec":"QEC Decode"}
DISPLAY = {"gemini-3.5-flash":"Gemini~3.5","claude-sonnet-5":"Sonnet~5","gpt-5.4":"GPT-5.4",
           "o4-mini":"o4-mini","gemma-4-31b":"Gemma-4","gpt-4.1-mini":"GPT-4.1m",
           "claude-opus":"Opus~4.1","gemini-flash-lite":"Gemini~FL",
           "mistral-large-3":"Mistral L3","llama-70b":"LLaMA-70B"}
A_MIN, A_MAX = 0.05, 6.0
LAMBDA = 4.0   # log-normal(0, 0.5) prior on discrimination


def load():
    """Return k[m, j] successes, n[m, j] trials, plus the model and item labels."""
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    models = sorted({r["model"] for r in rows})
    items = sorted({(r["task_name"], int(r["level"])) for r in rows})
    mi = {m: i for i, m in enumerate(models)}
    ji = {t: i for i, t in enumerate(items)}
    k = np.zeros((len(models), len(items)))
    n = np.zeros((len(models), len(items)))
    for r in rows:
        i, j = mi[r["model"]], ji[(r["task_name"], int(r["level"]))]
        n[i, j] += 1
        if str(r["correct"]).lower() == "true":
            k[i, j] += 1
    return k, n, models, items


def probs(theta, a, b):
    """p[m, j] under the 2PL."""
    eta = a[None, :] * (theta[:, None] - b[None, :])
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def fit(k, n, lam=LAMBDA, max_sweeps=500, tol=1e-8):
    M, J = k.shape
    # sensible starts: theta from model pass rate, b from item pass rate, a = 1
    pm = np.clip(k.sum(1) / n.sum(1), .02, .98)
    pj = np.clip(k.sum(0) / n.sum(0), .02, .98)
    theta = np.log(pm / (1 - pm))
    b = -np.log(pj / (1 - pj))
    a = np.ones(J)

    for sweep in range(max_sweeps):
        old = np.concatenate([theta, a, b])

        # --- abilities: 1-D Newton per model ---
        for m in range(M):
            for _ in range(25):
                p = 1 / (1 + np.exp(-np.clip(a * (theta[m] - b), -30, 30)))
                g = np.sum(a * (k[m] - n[m] * p))
                h = -np.sum(a**2 * n[m] * p * (1 - p))
                if abs(h) < 1e-12:
                    break
                step = np.clip(-g / h, -1.0, 1.0)
                theta[m] += step
                if abs(step) < 1e-10:
                    break

        # --- item parameters: 2-D Newton per item ---
        for j in range(J):
            for _ in range(25):
                z = theta - b[j]
                p = 1 / (1 + np.exp(-np.clip(a[j] * z, -30, 30)))
                w = n[:, j] * p * (1 - p)
                resid = k[:, j] - n[:, j] * p
                # gradient/Hessian of the penalised log-likelihood
                pen_g = -lam * math.log(a[j]) / a[j]
                pen_h = -lam * (1.0 - math.log(a[j])) / a[j]**2
                g = np.array([np.sum(resid * z) + pen_g, -a[j] * np.sum(resid)])
                H = np.array([
                    [-np.sum(w * z**2) + pen_h,            np.sum(a[j] * w * z - resid)],
                    [np.sum(a[j] * w * z - resid),         -a[j]**2 * np.sum(w)],
                ])
                if abs(np.linalg.det(H)) < 1e-12:
                    break
                d = np.clip(np.linalg.solve(H, -g), -0.5, 0.5)
                a[j] = float(np.clip(a[j] + d[0], A_MIN, A_MAX))
                b[j] = float(np.clip(b[j] + d[1], -8, 8))
                if np.max(np.abs(d)) < 1e-10:
                    break

        # --- rescale theta to mean 0, SD 1 (fixes the 2PL indeterminacy) ---
        mu, sd = theta.mean(), theta.std(ddof=0)
        if sd > 1e-8:
            theta = (theta - mu) / sd
            b = (b - mu) / sd
            a = a * sd

        if np.max(np.abs(np.concatenate([theta, a, b]) - old)) < tol:
            break
    return theta, a, b, sweep + 1


def standard_errors(theta, a, b, k, n, lam=LAMBDA):
    """Conditional observed-information SEs."""
    p = probs(theta, a, b)
    w = n * p * (1 - p)
    se_theta = 1.0 / np.sqrt(np.maximum((a[None, :] ** 2 * w).sum(1), 1e-12))
    se_a = np.full(len(a), np.nan)
    se_b = np.full(len(b), np.nan)
    for j in range(len(a)):
        z = theta - b[j]
        resid = k[:, j] - n[:, j] * p[:, j]
        pen_h = -lam * (1.0 - math.log(a[j])) / a[j]**2
        H = np.array([
            [-np.sum(w[:, j] * z**2) + pen_h,         np.sum(a[j] * w[:, j] * z - resid)],
            [np.sum(a[j] * w[:, j] * z - resid),      -a[j]**2 * np.sum(w[:, j])],
        ])
        if abs(np.linalg.det(H)) > 1e-12:
            C = np.linalg.inv(-H)
            if C[0, 0] > 0: se_a[j] = math.sqrt(C[0, 0])
            if C[1, 1] > 0: se_b[j] = math.sqrt(C[1, 1])
    return se_theta, se_a, se_b


def loglik(theta, a, b, k, n):
    p = np.clip(probs(theta, a, b), 1e-12, 1 - 1e-12)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser(description="2PL IRT over 55 (task, level) items")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--lam", type=float, default=LAMBDA,
                    help="log-normal prior strength on discrimination (0 = unpenalised)")
    args = ap.parse_args()

    k, n, models, items = load()
    theta, a, b, sweeps = fit(k, n, lam=args.lam)
    se_t, se_a, se_b = standard_errors(theta, a, b, k, n, lam=args.lam)
    ll = loglik(theta, a, b, k, n)
    npar = len(theta) + 2 * len(a)

    print("=" * 76)
    print("2PL IRT   P(correct) = 1 / (1 + exp(-a_j (theta_m - b_j)))")
    print("=" * 76)
    print(f"models {len(models)}   items {len(items)}   observations {int(n.sum())}   "
          f"obs/item {int(n.sum(0)[0])}")
    print(f"parameters {npar} = {len(theta)} theta + {len(a)} a + {len(b)} b")
    clamped = int(np.sum((a <= A_MIN + 1e-9) | (a >= A_MAX - 1e-9)))
    conv = "converged" if sweeps < 500 else "DID NOT CONVERGE (hit sweep cap)"
    print(f"prior on a: lambda = {args.lam}"
          f"{' (log-normal(0, %.2f))' % (1/math.sqrt(args.lam)) if args.lam > 0 else ' (unpenalised)'}")
    print(f"{conv} in {sweeps} sweeps   logLik {ll:.1f}   AIC {2*npar - 2*ll:.1f}")
    print(f"discriminations pinned at a search bound: {clamped}/{len(a)}")

    deg = [(items[j], k[:, j].sum(), n[:, j].sum()) for j in range(len(items))
           if k[:, j].sum() in (0, n[:, j].sum())]
    print(f"degenerate items (all-right or all-wrong across all models): {len(deg)}")
    for (t, l), s, tot in deg:
        print(f"    {TASK_LABEL[t]} L{l}: {int(s)}/{int(tot)}")

    print("\n--- MODEL ABILITY (theta, standardised to mean 0, SD 1) ---")
    print(f"  {'rank':<5}{'model':20}{'theta':>8}{'SE':>7}{'95% CI':>18}")
    order = np.argsort(-theta)
    for r, m in enumerate(order, 1):
        e, s = theta[m], se_t[m]
        print(f"  {r:<5}{models[m]:20}{e:+8.3f}{s:7.3f}   [{e-1.96*s:+.3f}, {e+1.96*s:+.3f}]")

    print("\n  adjacent pairs with overlapping CIs:")
    ov = False
    for r in range(len(order) - 1):
        m1, m2 = order[r], order[r + 1]
        d = theta[m1] - theta[m2]
        sd = math.sqrt(se_t[m1] ** 2 + se_t[m2] ** 2)
        if abs(d) < 1.96 * sd:
            print(f"    {models[m1]} vs {models[m2]}: diff {d:+.3f} +/- {1.96*sd:.3f}")
            ov = True
    if not ov:
        print("    none")

    print("\n--- ITEM PARAMETERS (55 items, sorted hardest first) ---")
    print(f"  {'item':24}{'b':>8}{'SE':>7}{'a':>8}{'SE':>7}{'pass':>7}")
    for j in np.argsort(-b):
        t, l = items[j]
        pr = k[:, j].sum() / n[:, j].sum()
        print(f"  {TASK_LABEL[t]+' L'+str(l):24}{b[j]:+8.3f}{se_b[j]:7.3f}"
              f"{a[j]:8.3f}{se_a[j]:7.3f}{pr:7.2f}")

    print(f"\n  b range {b.min():+.3f} .. {b.max():+.3f}   (span {b.max()-b.min():.2f})")
    print(f"  a range {a.min():.3f} .. {a.max():.3f}   median {np.median(a):.3f}")
    print(f"  items with a > 1.0: {int((a > 1.0).sum())}/{len(a)}")
    print(f"  theta range {theta.min():+.3f} .. {theta.max():+.3f}")
    print(f"  items harder than the ablest model: {int((b > theta.max()).sum())}/{len(b)}")

    if args.latex:
        emit_latex(theta, se_t, a, se_a, b, se_b, k, n, models, items, ll, npar)


def _n(x, d=3):
    t = f"{abs(x):.{d}f}"
    return f"$-${t}" if x < 0 else t


def emit_latex(theta, se_t, a, se_a, b, se_b, k, n, models, items, ll, npar):
    """One table, two stacked sub-tables, classic ruled grid (no booktabs)."""
    by_task = {}
    for j, (t, l) in enumerate(items):
        degenerate = k[:, j].sum() in (0, n[:, j].sum())
        by_task.setdefault(t, []).append((b[j], a[j], degenerate))
    agg = []
    for t, vals in by_task.items():
        good = [(bb, aa) for bb, aa, d in vals if not d]
        agg.append((TASK_LABEL[t], sum(x for x, _ in good) / len(good),
                    sum(x for _, x in good) / len(good), len(good)))
    agg.sort(key=lambda x: -x[1])

    print("\n" + "=" * 76 + "\nLATEX\n" + "=" * 76)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{2PL IRT fit. Items are the %d (task, level) cells, each administered to "
          "every model under %d seeds; abilities are standardised to mean 0 and unit SD. "
          "%d parameters (%d abilities, %d difficulties, %d discriminations) over %d responses. "
          "Discrimination carries a log-normal(0,\\,0.5) prior, without which the %d slopes are "
          "not identifiable from %d models. In (b), $n$ is the number of identified items entering "
          "each mean: State Preparation L1 and L2 were passed by every model under every seed, "
          "leaving their difficulty unidentified, so they are excluded.}"
          % (len(items), int(n[0, 0]), npar, len(theta), len(b), len(a), int(n.sum()),
             len(a), len(theta)))
    print("\\label{tab:irt2pl}")
    print("\\footnotesize")
    print("\\renewcommand{\\arraystretch}{1.15}")
    print()
    print("{\\smallskip (a) Model ability\\par\\smallskip}")
    print("\\begin{tabular}{|c|l|r|r|r|}")
    print("\\hline")
    print("\\textbf{Rank} & \\textbf{Model} & $\\theta$ & \\textbf{SE} "
          "& \\textbf{95\\% CI} \\\\")
    print("\\hline\\hline")
    for r, m in enumerate(np.argsort(-theta), 1):
        e, se = theta[m], se_t[m]
        print(f"{r} & {DISPLAY[models[m]]} & {_n(e)} & {se:.3f} & "
              f"[{_n(e-1.96*se)}, {_n(e+1.96*se)}] \\\\")
        print("\\hline")
    print("\\end{tabular}")
    print()
    print("\\vspace{2.5ex}")
    print()
    print("{\\smallskip (b) Task difficulty and discrimination\\par\\smallskip}")
    print("\\begin{tabular}{|l|r|r|c|}")
    print("\\hline")
    print("\\textbf{Task} & \\textbf{Mean difficulty} $\\bar{b}$ "
          "& \\textbf{Mean discrimination} $\\bar{a}$ & $n$ \\\\")
    print("\\hline\\hline")
    for nmz, mb, ma, cnt in agg:
        print(f"{nmz} & {_n(mb)} & {ma:.3f} & {cnt} \\\\")
        print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
