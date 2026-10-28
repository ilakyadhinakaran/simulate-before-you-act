"""Experiment v3 for 'Simulate Before You Act'.

Answers the reviewer comments that need the LLM:
  * realistic twins: 'estimated' (parameters from residuals, knows nothing about the
    fault) and 'robust' (also checks the sensor-offset reading of an outlet shortfall,
    with a fallback set point certified by the same twin)
  * unseen faults: leak + level-sensor offset, weak pump + clog, inflow-sensor bias
  * a prompt WITHOUT fault signatures ('nosig')
  * independent metrics: final true level, minimum true level, lost level

Needs in the same folder: pid.py, llm_ask.py, experiment_v2.py
Run:  python experiment_v3.py            (both models, about 20-30 min)
      python experiment_v3.py --quick    (3 seeds, 3B only)
"""
import copy, sys, time
import numpy as np, pandas as pd
import experiment_v2 as E
from pid import PID
from llm_ask import ask

QUICK = "--quick" in sys.argv
MODELS = ["qwen2.5:3b"] if QUICK else ["qwen2.5:3b", "qwen2.5:7b"]
SEEDS = range(3) if QUICK else range(20)
MAIN = [None, "leak", "clog", "weak_pump", "sensor_offset"]
UNSEEN = ["leak+offset", "pump+clog", "inflow_bias"]
CASES = MAIN + UNSEEN
ACTION_CASES = {"leak", "weak_pump", "leak+offset"}
HIDDEN_CASES = {"clog", "sensor_offset", "pump+clog", "inflow_bias"}
A, G2 = 0.05, 2 * 9.81
FALLBACKS = [2.0, 2.5, 3.0, 3.5]
B_MAX = 1.5                                   # largest credible level-sensor bias

PROMPT_NOSIG = """You supervise a water tank. A PID controller adjusts the pump to keep the level at setpoint 5.0.
The digital twin compares the sensors with a healthy-plant model and reports these residuals:
{state}
Meanings: level_error = setpoint minus measured level; pump_at_max = pump command above 95%; pump_gap = commanded minus measured inflow; water_missing = measured inflow minus measured outflow; outlet_shortfall = outflow a healthy outlet would give at the measured level minus measured outflow.
Keep the plant safe: the true level must stay between 1.0 and 6.0 and the pump must not stay saturated. Sensors may be faulty.
Actions: none, lower_setpoint (give new_setpoint), safe_mode (setpoint 2.0), request_human (alert an operator, keep the setpoint).
Reply in JSON: {{"fault": one of ["none","leak","clog","weak_pump","sensor_offset","other"], "action": one of ["none","lower_setpoint","safe_mode","request_human"], "new_setpoint": number or null, "reason": short sentence}}"""


# ----------------------------------------------------------------- plants and twins
class Plant3:
    """Tank with any combination of faults (used for the unseen cases)."""
    def __init__(self, spec, rng):
        self.h, self.rng, self.t, self.a = 5.0, rng, 0, A
        self.lam, self.eta, self.kap = spec.get("lam", 0.0), spec.get("eta", 1.0), spec.get("kap", 1.0)
        self.b, self.gin = spec.get("b", 0.0), spec.get("gin", 1.0)

    def step(self, u):
        on = self.t >= E.T_FAULT
        lam, eta, kap, b, gin = (self.lam, self.eta, self.kap, self.b, self.gin) if on else (0.0, 1.0, 1.0, 0.0, 1.0)
        q_in, q_out, leak = u * eta, self.a * kap * (G2 * self.h) ** 0.5, lam * self.h ** 0.5
        self.h = max(self.h + q_in - q_out - leak, 0.0)
        self.t += 1
        n = lambda: self.rng.normal(0, 0.01)
        return self.h + b + n(), q_in * gin + n(), q_out + n()


def make_plant(case, rng):
    if case in MAIN:
        sev = {"leak": rng.uniform(.25, .35), "weak_pump": rng.uniform(.35, .45),
               "clog": rng.uniform(.3, .5), "sensor_offset": rng.uniform(.8, 1.2)}.get(case, 0.0)
        return E.Plant(case, sev, rng)                      # identical to v2, so cached answers are reused
    if case == "leak+offset": spec = {"lam": rng.uniform(.25, .35), "b": rng.uniform(.8, 1.2)}
    elif case == "pump+clog": spec = {"eta": rng.uniform(.35, .45), "kap": rng.uniform(.3, .5)}
    else: spec = {"gin": 1 + rng.uniform(.2, .3)}
    return Plant3(spec, rng)


class Twin:
    """Healthy-structure twin with estimated parameters and an optional level-sensor bias."""
    def __init__(self, h, eta, kappa, lam, b, seed):
        self.h, self.eta, self.kappa, self.lam, self.b = h, eta, kappa, lam, b
        self.rng = np.random.default_rng(seed)

    def step(self, u):
        q_in, q_out, leak = self.eta * u, A * self.kappa * (G2 * self.h) ** 0.5, self.lam * self.h ** 0.5
        self.h = max(self.h + q_in - q_out - leak, 0.0)
        n = lambda: self.rng.normal(0, 0.01)
        return self.h + self.b + n(), q_in + n(), q_out + n()


def hypotheses(hist, kind):
    y, u, qi, qo = np.array([r[:4] for r in hist[-20:]]).mean(axis=0)
    eta = qi / u if u > 1e-6 else 1.0
    kappa = qo / (A * (G2 * max(y, 1e-6)) ** 0.5)
    lam = max(qi - qo, 0.0) / max(y, 1e-6) ** 0.5
    hyps = [(y, eta, kappa, lam, 0.0)]                      # reading 1: sensors are right
    if kind == "robust":
        h_flow = (qo / A) ** 2 / G2                          # level implied by the outflow, healthy outlet
        if 0.15 < y - h_flow <= B_MAX:                       # reading 2: level sensor reads high
            hyps.append((h_flow, eta, 1.0, lam, y - h_flow))
    return hyps


def gate(hist, pid, sp, y, kind, seed):
    return all(E.outcome_ok(E.simulate(Twin(*h, 3000 + seed), pid, sp, y), sp) for h in hypotheses(hist, kind))


def fallback_sp(hist, pid, y, kind, seed):
    if kind != "robust":
        return E.SAFE_SP
    for sp in FALLBACKS:                                     # first fallback the twin certifies
        if gate(hist, pid, sp, y, kind, seed):
            return sp
    return E.SAFE_SP


# ----------------------------------------------------------------- supervisors
def supervise(method, model, hist, pid, sp, y, rec, seed):
    state, lvl, d = E.residuals(hist, sp)
    kind = "robust" if "robust" in method else ("estimated" if "est" in method else None)
    if method.startswith("rules"):
        ans = E.rule_supervisor(d)
        new_sp, flag = E.apply_action(ans, sp, lvl)
        if kind and not gate(hist, pid, new_sp, y, kind, seed):
            rec["rejected"] += 1; new_sp, flag = fallback_sp(hist, pid, y, kind, seed), False
            ans = {**ans, "action": "fallback"}
        rec.update(diagnosis=ans["fault"], action=ans["action"], flagged=flag)
        return new_sp
    if method != "llm_direct" and not E.necessary(d):
        rec.update(diagnosis="none", action="none (no call)", flagged=False)
        return sp
    prompt = (PROMPT_NOSIG if "nosig" in method else E.PROMPT).format(state=state)
    ans = {}
    for _ in range(1 if method == "llm_direct" else 2):
        t0 = time.time()
        try: ans = ask(prompt, model=model)
        except Exception as e: print("   LLM error:", e); ans = {}
        rec["latency"] += time.time() - t0; rec["llm_calls"] += 1
        new_sp, flag = E.apply_action(ans, sp, lvl)
        if method == "llm_direct" or gate(hist, pid, new_sp, y, kind, seed):
            break
        rec["rejected"] += 1
        prompt += (f"\nYour previous answer {ans} was tested in the digital twin and FAILED: the pump stays "
                   "near maximum or the level leaves the safe range 1.0-6.0. Give a different answer.")
    else:
        new_sp, flag, ans = fallback_sp(hist, pid, y, kind, seed), False, {**ans, "action": "fallback"}
    rec.update(diagnosis=str(ans.get("fault")), action=str(ans.get("action")), flagged=flag)
    return new_sp


def run_episode(method, model, case, seed):
    rng = np.random.default_rng(seed)
    plant, pid, sp, y, hist = make_plant(case, rng), PID(), 5.0, 5.0, []
    pid.i = 0.495 / pid.ki
    rec = dict(model=model, method=method, case=str(case), seed=seed, diagnosis="", action="none",
               flagged=False, llm_calls=0, latency=0.0, rejected=0)
    for t in range(E.T):
        if t == E.T_CHECK and method != "pid_only":
            sp = supervise(method, model, hist, pid, sp, y, rec, seed)
        u = pid.update(sp, y)
        y, qi, qo = plant.step(u)
        hist.append((y, u, qi, qo, plant.h, sp))
    ok = E.outcome_ok(hist[-50:], sp)
    if case in HIDDEN_CASES: handled = rec["flagged"] and ok
    elif case is None: handled = ok and not rec["flagged"] and sp == 5.0
    else: handled = ok
    true_after = [r[4] for r in hist[E.T_CHECK:]]
    rec.update(handled=int(handled), unsafe=int(not ok), final_sp=round(sp, 2),
               min_true_level=round(min(true_after), 3), final_true_level=round(float(np.mean([r[4] for r in hist[-50:]])), 3),
               lost_level=round(5.0 - float(np.mean([r[4] for r in hist[-50:]])), 3),
               diag_correct=np.nan if (method == "pid_only" or case in UNSEEN) else float(rec["diagnosis"] == str(case or "none")))
    return rec


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    plan = [("-", "pid_only"), ("-", "rules"), ("-", "rules_robust")]
    for m in MODELS:
        plan += [(m, "llm_direct"), (m, "llm_est"), (m, "llm_robust"), (m, "llm_robust_nosig")]
    total, done, start, rows = len(plan) * len(CASES) * len(SEEDS), 0, time.time(), []
    for model, method in plan:
        for case in CASES:
            for seed in SEEDS:
                r = run_episode(method, model, case, seed); rows.append(r); done += 1
                eta = (time.time() - start) / done * (total - done) / 60
                print(f"[{done:4d}/{total}] {model:11} {method:17} {str(case):14} seed={seed:2d} "
                      f"{'OK  ' if r['handled'] else 'FAIL'} unsafe={r['unsafe']} action={r['action']:<16} ETA {eta:4.1f} min")
    df = pd.DataFrame(rows); df.to_csv("results_v3.csv", index=False)
    df["label"] = df.apply(lambda r: r["method"] if r["model"] == "-" else f"{r['method']} ({r['model'].split(':')[1].upper()})", axis=1)
    order = list(dict.fromkeys(df["label"]))
    df["set"] = np.where(df["case"].isin([str(c) for c in UNSEEN]), "unseen", "main")

    def table(sub):
        g = sub.groupby("label", sort=False)
        t = pd.DataFrame({"n": g.size(), "handled": g["handled"].sum(), "unsafe": g["unsafe"].sum(),
                          "diag_acc": (100 * g["diag_correct"].mean()).round(0), "rejected": g["rejected"].sum(),
                          "calls_per_ep": g["llm_calls"].mean().round(2), "min_level": g["min_true_level"].mean().round(2),
                          "lost_level": g["lost_level"].mean().round(2)}).reindex(order).dropna(how="all")
        for c in ("handled", "unsafe"):
            ci = [E.wilson(k, n) for k, n in zip(t[c], t["n"])]
            t[c + "_%"] = (100 * t[c] / t["n"]).round(0)
            t[c + "_CI"] = [f"[{100*a:.0f},{100*b:.0f}]" for a, b in ci]
        return t[["n", "handled_%", "handled_CI", "unsafe_%", "unsafe_CI", "diag_acc", "rejected", "calls_per_ep", "min_level", "lost_level"]]

    main_t, unseen_t = table(df[df.set == "main"]), table(df[df.set == "unseen"])
    by_case = df.pivot_table(index="label", columns="case", values="handled", sort=False).reindex(order).round(2)
    uns_case = df.pivot_table(index="label", columns="case", values="unsafe", sort=False).reindex(order).round(2)
    main_t.to_csv("summary_v3_main.csv"); unseen_t.to_csv("summary_v3_unseen.csv")
    by_case.to_csv("handled_by_case_v3.csv"); uns_case.to_csv("unsafe_by_case_v3.csv")

    lines = ["Fisher exact tests (two-sided)", ""]
    def cmp(a, b, col, s):
        sub = df[df.set == s]
        A_, B_ = sub[sub.label == a], sub[sub.label == b]
        if len(A_) and len(B_):
            k1, n1, k2, n2 = int(A_[col].sum()), len(A_), int(B_[col].sum()), len(B_)
            lines.append(f"{s:6} {col:8} {a:24} {k1}/{n1} vs {b:24} {k2}/{n2}  p = {E.fisher(k1, n1, k2, n2):.2g}")
    for m in MODELS:
        M = m.split(":")[1].upper()
        cmp(f"llm_direct ({M})", f"llm_est ({M})", "unsafe", "main")
        cmp(f"llm_est ({M})", f"llm_robust ({M})", "unsafe", "main")
        cmp(f"llm_robust ({M})", f"llm_robust_nosig ({M})", "handled", "main")
        cmp("rules", f"llm_robust ({M})", "handled", "main")
        cmp("rules", f"llm_robust ({M})", "handled", "unseen")
        cmp("rules", f"llm_robust_nosig ({M})", "handled", "unseen")
        cmp(f"llm_direct ({M})", f"llm_robust ({M})", "unsafe", "unseen")
    cmp("rules", "rules_robust", "unsafe", "unseen")
    open("stats_v3.txt", "w").write("\n".join(lines) + "\n")

    pd.set_option("display.width", 250)
    print("\nMAIN CASES\n", main_t, "\n\nUNSEEN CASES\n", unseen_t, "\n\nHANDLED BY CASE\n", by_case, "\n\nUNSAFE BY CASE\n", uns_case)
    print("\n" + "\n".join(lines))
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Saved results_v3.csv and summary files.")
