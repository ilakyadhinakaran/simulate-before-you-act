"""Gate study: how well does the twin check separate safe from unsafe actions
when the twin does NOT know the fault?

No LLM is needed: actions come from a random proposer, so the result describes
the gate itself. For every episode (5 cases x 20 seeds) the plant runs under PID
until t = 180; then 12 candidate setpoints are proposed. Each one is
  * judged by the gate, using one of several twins, and
  * applied to the TRUE plant for the remaining 220 steps (ground truth).

Twins:
  oracle      : exact copy of the faulty plant (what the v1/v2 experiments used)
  estimated   : healthy-plant model whose leak, pump-efficiency and outlet
                parameters are estimated from the residuals; starts from the
                MEASURED level; own noise; knows nothing about the fault
  est+10/20/30: estimated twin with every parameter perturbed by 10/20/30 %

Ground truth uses an independent criterion over the WHOLE remaining run:
  unsafe if the true level ever leaves [1.0, 6.0] or the pump stays
  saturated (mean command > 0.95 over the last 20 steps), or the level
  fails to reach the new setpoint (|mean measured - sp| > 0.15).
"""
import copy, sys
import numpy as np, pandas as pd
import experiment_v2 as E
from pid import PID

A, G2 = 0.05, 2 * 9.81


class TwinPlant:
    """Healthy-structure tank model with estimated parameters."""
    def __init__(self, h, eta, kappa, lam, seed):
        self.h, self.eta, self.kappa, self.lam = h, eta, kappa, lam
        self.rng = np.random.default_rng(seed)

    def step(self, u):
        q_in = self.eta * u
        q_out = A * self.kappa * (G2 * self.h) ** 0.5
        leak = self.lam * self.h ** 0.5
        self.h = max(self.h + q_in - q_out - leak, 0.0)
        n = lambda: self.rng.normal(0, 0.01)
        return self.h + n(), q_in + n(), q_out + n()


def estimate(hist):
    y, u, qi, qo = np.array([r[:4] for r in hist[-20:]]).mean(axis=0)
    eta = qi / u if u > 1e-6 else 1.0
    kappa = qo / (A * (G2 * max(y, 1e-6)) ** 0.5)
    lam = max(qi - qo, 0.0) / max(y, 1e-6) ** 0.5
    return y, eta, kappa, lam


def gate_ok(twin, pid, sp, y):
    return E.outcome_ok(E.simulate(twin, pid, sp, y), sp)


def truth_ok(plant, pid, sp, y, steps=220):
    rows = E.simulate(plant, pid, sp, y, steps)      # deepcopy: the real plant is not consumed
    return E.outcome_ok(rows, sp)


def run(seeds=range(20), n_props=12):
    out = []
    for fault in E.FAULTS:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sev = {"leak": rng.uniform(.25, .35), "weak_pump": rng.uniform(.35, .45),
                   "clog": rng.uniform(.3, .5), "sensor_offset": rng.uniform(.8, 1.2)}.get(fault, 0.0)
            plant, pid, y, hist = E.Plant(fault, sev, rng), PID(), 5.0, []
            pid.i = 0.495 / pid.ki
            for t in range(E.T_CHECK):
                u = pid.update(5.0, y); y, qi, qo = plant.step(u); hist.append((y, u, qi, qo, plant.h, 5.0))
            ym, eta, kappa, lam = estimate(hist)
            prng = np.random.default_rng(1000 + seed)
            props = [5.0, 2.0] + list(np.round(prng.uniform(0.5, 5.5, n_props - 2), 2))
            erng = np.random.default_rng(2000 + seed)
            errs = {e: [1 + erng.choice([-1, 1]) * e for _ in range(3)] for e in (0.1, 0.2, 0.3)}
            for sp in props:
                truth = truth_ok(plant, pid, sp, y)
                rec = dict(fault=str(fault), seed=seed, sp=sp, safe=truth)
                rec["oracle"] = gate_ok(plant, pid, sp, y)
                rec["estimated"] = gate_ok(TwinPlant(ym, eta, kappa, lam, 3000 + seed), pid, sp, y)
                for e, (m1, m2, m3) in errs.items():
                    tw = TwinPlant(ym, eta * m1, kappa * m2, lam * m3, 3000 + seed)
                    rec[f"est+{int(e*100)}%"] = gate_ok(tw, pid, sp, y)
                out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = run()
    df.to_csv("gate_study.csv", index=False)
    twins = ["oracle", "estimated", "est+10%", "est+20%", "est+30%"]
    unsafe, safe = df[~df.safe], df[df.safe]
    rows = []
    for tw in twins:
        fa = unsafe[tw].sum(); fr = (~safe[tw]).sum()
        rows.append(dict(twin=tw, proposals=len(df), truly_unsafe=len(unsafe), truly_safe=len(safe),
                         false_accept=int(fa), false_accept_rate=round(fa / len(unsafe), 3),
                         false_reject=int(fr), false_reject_rate=round(fr / len(safe), 3)))
    summ = pd.DataFrame(rows); summ.to_csv("gate_summary.csv", index=False)
    print(summ.to_string(index=False))
    print()
    byf = df.groupby("fault").apply(lambda d: pd.Series({tw: round(d[~d.safe][tw].mean(), 3) if (~d.safe).any() else np.nan for tw in twins} | {"n_unsafe": int((~d.safe).sum()), "n": len(d)}))
    print("False-accept rate by fault:"); print(byf.to_string())
    byf.to_csv("gate_by_fault.csv")
