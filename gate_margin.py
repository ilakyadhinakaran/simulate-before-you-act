"""Add a safety margin to the estimated-twin gate and measure the trade-off."""
import numpy as np, pandas as pd
import experiment_v2 as E, gate_study as GS
from pid import PID

def ok_margin(rows, sp, m):
    true = [r[4] for r in rows]; cmd = [r[1] for r in rows[-20:]]; meas = [r[0] for r in rows[-20:]]
    return min(true) >= 1.0 + m and max(true) <= 6.0 - m and np.mean(cmd) < 0.95 - 0.1 * m and abs(np.mean(meas) - sp) < 0.15

out = []
for fault in E.FAULTS:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        sev = {"leak": rng.uniform(.25, .35), "weak_pump": rng.uniform(.35, .45),
               "clog": rng.uniform(.3, .5), "sensor_offset": rng.uniform(.8, 1.2)}.get(fault, 0.0)
        plant, pid, y, hist = E.Plant(fault, sev, rng), PID(), 5.0, []
        pid.i = 0.495 / pid.ki
        for t in range(E.T_CHECK):
            u = pid.update(5.0, y); y, qi, qo = plant.step(u); hist.append((y, u, qi, qo, plant.h, 5.0))
        ym, eta, kappa, lam = GS.estimate(hist)
        prng = np.random.default_rng(1000 + seed)
        props = [5.0, 2.0] + list(np.round(prng.uniform(0.5, 5.5, 10), 2))
        for sp in props:
            truth = GS.truth_ok(plant, pid, sp, y)
            rows = E.simulate(GS.TwinPlant(ym, eta, kappa, lam, 3000 + seed), pid, sp, y)
            rec = dict(fault=str(fault), seed=seed, sp=sp, safe=truth)
            for m in (0.0, 0.5, 1.0, 1.5):
                rec[f"m{m}"] = ok_margin(rows, sp, m)
            out.append(rec)
df = pd.DataFrame(out)
uns, saf = df[~df.safe], df[df.safe]
res = []
for m in (0.0, 0.5, 1.0, 1.5):
    c = f"m{m}"
    res.append(dict(margin=m, false_accept_rate=round(uns[c].mean(), 3), false_reject_rate=round((~saf[c]).mean(), 3),
                    fa_sensor_offset=round(uns[uns.fault == "sensor_offset"][c].mean(), 3)))
r = pd.DataFrame(res); r.to_csv("gate_margin.csv", index=False); print(r.to_string(index=False))
