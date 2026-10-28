"""Experiment v2 for 'Simulate Before You Act'.

New in v2:
  * 20 seeds per case (seeds 0-4 reuse the cached LLM answers from v1)
  * rules        : rule-based supervisor using the same fault signatures (no LLM)
  * llm_nec      : LLM + twin check + necessity check (LLM only called if a residual is significant)
  * llm_mismatch : LLM + twin check, but the twin has 10-20% model error
  * Fisher exact tests and Wilson 95% CIs
Outputs: results_v2.csv, summary_v2.csv, handled_by_fault_v2.csv, stats_v2.txt,
         fig_rates_v2.png, fig_handled_v2.png
Run:  python experiment_v2.py            (both models)
      python experiment_v2.py --quick    (3 seeds, 3B only - a 2-minute test run)
"""
import copy, math, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pid import PID
from llm_ask import ask

QUICK = "--quick" in sys.argv
MODELS = ["qwen2.5:3b"] if QUICK else ["qwen2.5:3b", "qwen2.5:7b"]
SEEDS = range(3) if QUICK else range(20)
FAULTS = [None, "leak", "clog", "weak_pump", "sensor_offset"]
LLM_METHODS = ["llm_direct", "llm_checked", "llm_nec", "llm_mismatch"]
SP0, T, T_FAULT, T_CHECK, SAFE_SP = 5.0, 400, 100, 180, 2.0
G = 0.05 * (2 * 9.81) ** 0.5            # healthy outflow = G * sqrt(level)
SIG = 0.03                               # residual significance threshold

PROMPT = """You supervise a water tank. A PID controller adjusts the pump to keep the level at setpoint 5.0.
The digital twin compares the sensors with a healthy-plant model and reports these residuals:
{state}
A residual above 0.03 is significant. Fault signatures:
- none: all residuals near 0
- weak_pump: pump_gap significant (pump delivers less than commanded)
- leak: water_missing significant (more water goes in than comes out)
- clog: outlet_shortfall above 0.2
- sensor_offset: outlet_shortfall between 0.03 and 0.2
Actions:
- If level_error is large and pump_at_max is True, PID cannot recover: choose lower_setpoint with a new_setpoint safely below the current level, or safe_mode.
- If level_error is near 0 but a fault is present, choose request_human.
- Otherwise choose none.
Reply in JSON: {{"fault": one of ["none","leak","clog","weak_pump","sensor_offset"], "action": one of ["none","lower_setpoint","safe_mode","request_human"], "new_setpoint": number or null, "reason": short sentence}}"""


class Plant:
    def __init__(self, fault, sev, rng):
        self.h, self.fault, self.sev, self.rng, self.t = 5.0, fault, sev, rng, 0
        self.a = 0.05                    # outlet coefficient (a twin with model error gets a wrong value)

    def step(self, u):
        f, s = (self.fault if self.t >= T_FAULT else None), self.sev
        q_in = u * (s if f == "weak_pump" else 1.0)
        q_out = self.a * (s if f == "clog" else 1.0) * (2 * 9.81 * self.h) ** 0.5
        leak = s * self.h ** 0.5 if f == "leak" else 0.0
        self.h = max(self.h + q_in - q_out - leak, 0.0)
        self.t += 1
        n = lambda: self.rng.normal(0, 0.01)
        meas = self.h + (s if f == "sensor_offset" else 0.0)
        return meas + n(), q_in + n(), q_out + n()


def residuals(hist, sp):
    lvl, cmd, qi, qo = np.array([r[:4] for r in hist[-20:]]).mean(axis=0)
    d = dict(level_error=sp - lvl, current_level=lvl, pump_at_max=cmd > 0.95,
             pump_gap=cmd - qi, water_missing=qi - qo, outlet_shortfall=G * max(lvl, 0) ** 0.5 - qo)
    text = (f"level_error={d['level_error']:.3f}, current_level={lvl:.2f}, pump_at_max={d['pump_at_max']}, "
            f"pump_gap={d['pump_gap']:.3f}, water_missing={d['water_missing']:.3f}, "
            f"outlet_shortfall={d['outlet_shortfall']:.3f}")
    return text, lvl, d


def necessary(d):
    """Necessity check: act only if some residual shows a problem."""
    return (abs(d["level_error"]) > 0.1 or d["pump_at_max"] or d["pump_gap"] > SIG
            or d["water_missing"] > SIG or d["outlet_shortfall"] > SIG)


def rule_supervisor(d):
    """Same fault signatures and action rules as the prompt, written as code."""
    if d["pump_gap"] > SIG: fault = "weak_pump"
    elif d["water_missing"] > SIG: fault = "leak"
    elif d["outlet_shortfall"] > 0.2: fault = "clog"
    elif d["outlet_shortfall"] > SIG: fault = "sensor_offset"
    else: fault = "none"
    if d["level_error"] > 0.1 and d["pump_at_max"]: action = "safe_mode"
    elif fault != "none": action = "request_human"
    else: action = "none"
    return {"fault": fault, "action": action, "new_setpoint": None}


def apply_action(ans, sp, lvl):
    a = ans.get("action")
    if a == "lower_setpoint":
        try: return float(ans.get("new_setpoint")), False
        except (TypeError, ValueError): return lvl - 0.5, False
    if a == "safe_mode": return SAFE_SP, False
    return sp, a == "request_human"


def outcome_ok(rows, sp):
    true, cmd, meas = [r[4] for r in rows], [r[1] for r in rows[-20:]], [r[0] for r in rows[-20:]]
    return min(true) >= 1.0 and max(true) <= 6.0 and np.mean(cmd) < 0.95 and abs(np.mean(meas) - sp) < 0.15


def simulate(twin, pid, sp, y, steps=150):
    p, c, rows = copy.deepcopy(twin), copy.deepcopy(pid), []
    for _ in range(steps):
        u = c.update(sp, y); y, qi, qo = p.step(u); rows.append((y, u, qi, qo, p.h, sp))
    return rows


def make_twin(plant, method, seed):
    twin = copy.deepcopy(plant)
    if method == "llm_mismatch":                 # 10-20% error in outlet coefficient and fault severity
        r = np.random.default_rng(10_000 + seed)
        twin.a *= 1 + r.choice([-1, 1]) * r.uniform(0.10, 0.20)
        twin.sev *= 1 + r.choice([-1, 1]) * r.uniform(0.10, 0.20)
    return twin


def supervise(method, plant, pid, sp, hist, rec, model, seed):
    state, lvl, d = residuals(hist, sp)
    if method == "rules":
        ans = rule_supervisor(d)
        new_sp, flag = apply_action(ans, sp, lvl)
        rec.update(diagnosis=ans["fault"], action=ans["action"], flagged=flag)
        return new_sp
    if method == "llm_nec" and not necessary(d):
        rec.update(diagnosis="none", action="none (no LLM call)", flagged=False)
        return sp
    prompt, y = PROMPT.format(state=state), hist[-1][0]
    checked = method != "llm_direct"
    ans = {}
    for _ in range(2 if checked else 1):
        t0 = time.time()
        try: ans = ask(prompt, model=model)
        except Exception as e: print("   LLM error:", e); ans = {}
        rec["latency"] += time.time() - t0; rec["llm_calls"] += 1
        new_sp, flag = apply_action(ans, sp, lvl)
        if not checked or outcome_ok(simulate(make_twin(plant, method, seed), pid, new_sp, y), new_sp):
            break
        rec["rejected"] += 1
        prompt += (f"\nYour previous answer {ans} was tested in the digital twin and FAILED: the pump stays "
                   "near maximum or the level leaves the safe range 1.0-6.0. Give a different answer.")
    else:
        new_sp, flag, ans = SAFE_SP, False, {**ans, "action": "fallback_safe_mode"}
    rec.update(diagnosis=str(ans.get("fault")), action=str(ans.get("action")), flagged=flag)
    return new_sp


def run_episode(method, fault, seed, model):
    rng = np.random.default_rng(seed)
    sev = {"leak": rng.uniform(.25, .35), "weak_pump": rng.uniform(.35, .45),
           "clog": rng.uniform(.3, .5), "sensor_offset": rng.uniform(.8, 1.2)}.get(fault, 0.0)
    plant, pid, sp, y, hist = Plant(fault, sev, rng), PID(), SP0, SP0, []
    pid.i = 0.495 / pid.ki
    rec = dict(model=model, method=method, fault=str(fault), seed=seed, severity=round(sev, 3),
               diagnosis="", action="none", flagged=False, llm_calls=0, latency=0.0, rejected=0)
    for t in range(T):
        if t == T_CHECK and method != "pid_only":
            sp = supervise(method, plant, pid, sp, hist, rec, model, seed)
        u = pid.update(sp, y)
        y, qi, qo = plant.step(u)
        hist.append((y, u, qi, qo, plant.h, sp))
    ok = outcome_ok(hist[-50:], sp)
    if fault in ("clog", "sensor_offset"): handled = rec["flagged"]
    elif fault is None: handled = ok and not rec["flagged"] and sp == SP0
    else: handled = ok
    rec.update(handled=int(handled), unsafe_end=int(not ok), final_sp=round(sp, 2),
               diag_correct=np.nan if method == "pid_only" else float(rec["diagnosis"] == str(fault or "none")))
    return rec


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact test for k1/n1 vs k2/n2."""
    K, N = k1 + k2, n1 + n2
    pr = lambda k: math.comb(n1, k) * math.comb(n2, K - k) / math.comb(N, K)
    p0 = pr(k1)
    return min(1.0, sum(pr(k) for k in range(max(0, K - n2), min(K, n1) + 1) if pr(k) <= p0 * (1 + 1e-9)))


if __name__ == "__main__":
    plan = [("-", "pid_only"), ("-", "rules")] + [(m, me) for m in MODELS for me in LLM_METHODS]
    total, done, start, rows = len(plan) * len(FAULTS) * len(SEEDS), 0, time.time(), []
    for model, method in plan:
        for fault in FAULTS:
            for seed in SEEDS:
                rec = run_episode(method, fault, seed, model)
                rows.append(rec); done += 1
                eta = (time.time() - start) / done * (total - done) / 60
                print(f"[{done:4d}/{total}] {model:11} {method:13} {str(fault):14} seed={seed:2d} "
                      f"{'OK  ' if rec['handled'] else 'FAIL'} action={rec['action']:<20} ETA {eta:4.1f} min")

    df = pd.DataFrame(rows); df.to_csv("results_v2.csv", index=False)
    df["label"] = df.apply(lambda r: r["method"] if r["model"] == "-" else f"{r['method']} ({r['model'].split(':')[1].upper()})", axis=1)
    order = list(dict.fromkeys(df["label"]))
    g = df.groupby("label", sort=False)
    summ = pd.DataFrame({
        "n": g.size(), "handled": g["handled"].sum(), "unsafe": g["unsafe_end"].sum(),
        "diag_acc": g["diag_correct"].mean().round(2), "rejections": g["rejected"].sum(),
        "llm_calls_per_ep": g["llm_calls"].mean().round(2)}).loc[order]
    summ["handled_%"] = (100 * summ.handled / summ.n).round(0)
    summ["handled_CI"] = [f"[{100*a:.0f}, {100*b:.0f}]" for a, b in (wilson(k, n) for k, n in zip(summ.handled, summ.n))]
    summ["unsafe_%"] = (100 * summ.unsafe / summ.n).round(0)
    summ["unsafe_CI"] = [f"[{100*a:.0f}, {100*b:.0f}]" for a, b in (wilson(k, n) for k, n in zip(summ.unsafe, summ.n))]
    summ.to_csv("summary_v2.csv")
    by_fault = df.pivot_table(index="label", columns="fault", values="handled", sort=False).loc[order].round(2)
    by_fault.to_csv("handled_by_fault_v2.csv")

    lines = ["Fisher exact tests (two-sided)", ""]
    def cmp(a, b, col):
        if a in summ.index and b in summ.index:
            k1, n1, k2, n2 = summ.loc[a, col], summ.loc[a, "n"], summ.loc[b, col], summ.loc[b, "n"]
            lines.append(f"{col:8} {a:22} {k1}/{n1}  vs  {b:22} {k2}/{n2}   p = {fisher(int(k1), int(n1), int(k2), int(n2)):.2g}")
    for m in MODELS:
        M = m.split(":")[1].upper()
        cmp(f"llm_direct ({M})", f"llm_checked ({M})", "unsafe")
        cmp(f"llm_direct ({M})", f"llm_checked ({M})", "handled")
        cmp("pid_only", f"llm_checked ({M})", "handled")
        cmp("rules", f"llm_checked ({M})", "handled")
        cmp("rules", f"llm_nec ({M})", "handled")
        cmp(f"llm_checked ({M})", f"llm_nec ({M})", "handled")
        cmp(f"llm_checked ({M})", f"llm_mismatch ({M})", "unsafe")
    open("stats_v2.txt", "w").write("\n".join(lines) + "\n")

    # Figure: handled and unsafe rates with 95% CIs
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(7, 0.35 * len(order) + 1.2), sharey=True)
    for ax, col, title in [(axs[0], "handled", "(a) Handled correctly"), (axs[1], "unsafe", "(b) Unsafe end state")]:
        for i, lab in enumerate(order):
            k, n = summ.loc[lab, col], summ.loc[lab, "n"]; lo, hi = wilson(k, n)
            c = "#2a78d6" if ("checked" in lab or "nec" in lab or "mismatch" in lab) else "#a9a7a1"
            ax.plot([100 * lo, 100 * hi], [i, i], color=c, lw=2); ax.plot(100 * k / n, i, "o", color=c, ms=5)
            ax.text(100 * hi + 3, i, f"{100*k/n:.0f}%", va="center", fontsize=8)
        ax.set_xlim(-3, 100); ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_title(title, loc="left", fontweight="bold")
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
    fig.tight_layout(); fig.savefig("fig_rates_v2.png", dpi=300); plt.close(fig)

    # Figure: handled rate per method and case
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(order) + 1.0))
    cols = ["None", "leak", "weak_pump", "clog", "sensor_offset"]
    mat = by_fault[cols].values
    ax.imshow(mat, cmap="Blues", vmin=-0.15, vmax=1.3, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100*mat[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["Healthy", "Leak", "Weak pump", "Clog", "Sensor offset"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig("fig_handled_v2.png", dpi=300); plt.close(fig)

    print("\n", summ[["n", "handled_%", "handled_CI", "unsafe_%", "unsafe_CI", "diag_acc", "rejections", "llm_calls_per_ep"]])
    print("\n", by_fault)
    print("\n" + "\n".join(lines))
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Saved: results_v2.csv, summary_v2.csv, "
          "handled_by_fault_v2.csv, stats_v2.txt, fig_rates_v2.png, fig_handled_v2.png")
