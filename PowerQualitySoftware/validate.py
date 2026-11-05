"""Validate the analyzer against synthetic signals with known harmonic content.

For every trial: build a waveform with signal_generator (true RMS / fundamental / THD known
analytically), run main.analyze() on it, and record the error. Results go to
results/validation.csv and a readable summary to results/summary.txt.

Run:
    python validate.py                 # full run (a few minutes)
    python validate.py --quick         # small run to check everything works
    python validate.py --seed 7        # different random trials (same seed -> identical trials)

Exit code 0 if every gated trial is within the gate limits, 1 otherwise (use --no-gate to skip).

Error definitions (decided BEFORE looking at results):
  THD error   : absolute, in percentage points  |THD_measured - THD_true|
                relative, in percent            100 * |THD_measured - THD_true| / THD_true
  f0 error    : |f_measured - f_true| in Hz, and in FFT bins (bin = fs / N)
  RMS error   : relative, in percent

Tiers:
  1M        1,048,575 samples at 10 kHz (the headline claim)
  100k      100,000 samples, fs varied (10 / 12.8 / 5.12 kHz)
  csv_e2e   one 1M-sample signal written to a CSV and read back through main.py's loader
  short     0.2 s records (about 10-12 cycles). Reported separately; NOT gated.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import main as analyzer
from signal_generator import generate_signal, harmonics_for_thd


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MASTER_SEED = 20251105
MAX_ORDER = 50                 # THD is evaluated over harmonics 2..50
NYQ_MARGIN = 0.95              # only use harmonics below 95% of Nyquist

N_1M = 1_048_575
N_100K = 100_000
SHORT_SECONDS = 0.2

THD_LEVELS = [0.5, 2.0, 5.0, 9.0, 20.0]              # true THD, percent
F0_VALUES = [50.0, 59.5, 59.93, 60.0, 60.4]          # Hz (60.0 is bin-aligned; others are not)
SNR_VALUES = [None, 60, 40, 30]                      # dB; None = noiseless
DC_VALUES = [0.0, 0.05]                              # volts, on a 1 V-peak fundamental
FS_VALUES = [10_000, 12_800, 5_120]
FS_WEIGHTS = [0.6, 0.2, 0.2]
ORDER_SETS = {
    "low_odd": [3, 5, 7],
    "mixed": [2, 3, 4, 5],
    "high": [29, 41, 49, 50],
    "full": list(range(2, MAX_ORDER + 1)),
}

GATED_TIERS = ("1M", "100k", "csv_e2e")

# Gate limits (worst case over gated trials). Calibrated from a full run: see summary.txt.
GATE_THD_PP = 0.05             # THD absolute error, percentage points
GATE_F0_BINS = 1.0             # fundamental error, in FFT bins
GATE_RMS_PCT = 0.05            # RMS relative error, percent
GATE_MIN_SNR_DB = 40           # trials noisier than this are reported, not gated

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Trial definition and execution
# ---------------------------------------------------------------------------
def make_params(tier, n, fs, f0, thd, order_set, snr, dc, random_phase):
    return dict(tier=tier, n=n, fs=fs, f0=f0, thd_true_target=thd, order_set=order_set,
                snr_db=snr, dc=dc, random_phase=random_phase)


def usable_orders(order_set, f0, fs):
    """Orders of this set that sit safely below Nyquist for this f0 / fs."""
    return [o for o in ORDER_SETS[order_set] if o * f0 <= NYQ_MARGIN * fs / 2]


def measure(row, v, truth, fs):
    """Run the analyzer on v and add measured values and error metrics to row."""
    n = len(v)
    t0 = time.perf_counter()
    try:
        r = analyzer.analyze(v, 1.0 / fs, fs, MAX_ORDER)
    except analyzer.AnalyzerError as e:
        row.update(status=f"analyzer error: {e}")
        return row
    row["analyze_s"] = time.perf_counter() - t0
    row["status"] = "ok"

    bin_res = fs / n
    row.update(
        thd_true=truth["thd"], thd_meas=r["thd"],
        thd_signed_err_pp=r["thd"] - truth["thd"],
        thd_abs_err_pp=abs(r["thd"] - truth["thd"]),
        thd_rel_err_pct=(100 * abs(r["thd"] - truth["thd"]) / truth["thd"]) if truth["thd"] > 0 else np.nan,
        f0_true=truth["fundamental_freq"], f0_meas=r["fundamental_freq"],
        f0_err_hz=abs(r["fundamental_freq"] - truth["fundamental_freq"]),
        f0_err_bins=abs(r["fundamental_freq"] - truth["fundamental_freq"]) / bin_res,
        rms_true=truth["rms"], rms_meas=r["rms"],
        rms_rel_err_pct=100 * abs(r["rms"] - truth["rms"]) / truth["rms"],
    )
    return row


def run_trial(p, seed):
    rng = np.random.default_rng(seed)
    orders = usable_orders(p["order_set"], p["f0"], p["fs"])
    row = dict(p, seed=seed, orders_used=len(orders))
    if not orders:
        row["status"] = "skipped: no harmonics below Nyquist"
        return row
    harmonics = harmonics_for_thd(p["thd_true_target"], orders, rng)
    _, v, truth = generate_signal(
        f0=p["f0"], fs=p["fs"], n_samples=p["n"], harmonics=harmonics, dc=p["dc"],
        snr_db=p["snr_db"], random_phase=p["random_phase"],
        seed=int(rng.integers(2 ** 31 - 1)), max_order=MAX_ORDER,
    )
    return measure(row, v, truth, p["fs"])


def run_csv_end_to_end(seed):
    """One 1M-sample signal -> CSV file -> main.py's loader -> analyzer."""
    p = make_params("csv_e2e", N_1M, 10_000, 59.93, 9.0, "low_odd", 50, 0.0, True)
    rng = np.random.default_rng(seed)
    h = harmonics_for_thd(p["thd_true_target"], usable_orders("low_odd", p["f0"], p["fs"]), rng)
    t, v, truth = generate_signal(f0=p["f0"], fs=p["fs"], n_samples=p["n"], harmonics=h,
                                  snr_db=p["snr_db"], seed=int(rng.integers(2 ** 31 - 1)))
    row = dict(p, seed=seed, orders_used=len(h))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "e2e.csv")
        pd.DataFrame({"t": t, "combine": v}).to_csv(path, index=False)
        try:
            t_in, v_in = analyzer.load_waveform(path, "t", "combine")
            _, fs_in = analyzer.check_sampling(t_in)
        except analyzer.AnalyzerError as e:
            row["status"] = f"loader error: {e}"
            return row
    return measure(row, v_in, truth, fs_in)


# ---------------------------------------------------------------------------
# Trial list: hand-picked corner cases + seeded random draws
# ---------------------------------------------------------------------------
def corner_cases():
    return [
        make_params("1M", N_1M, 10_000, 60.0, 9.0, "low_odd", None, 0.0, True),    # headline
        make_params("1M", N_1M, 10_000, 59.93, 5.0, "mixed", 40, 0.0, True),
        make_params("1M", N_1M, 10_000, 60.4, 0.5, "full", 40, 0.0, True),         # hard: tiny THD
        make_params("1M", N_1M, 10_000, 50.0, 20.0, "high", 60, 0.05, True),
        make_params("100k", N_100K, 10_000, 59.5, 0.5, "mixed", 30, 0.0, True),    # hard: noisy
        make_params("100k", N_100K, 10_000, 60.0, 2.0, "full", 30, 0.0, False),    # zero phase
        make_params("100k", N_100K, 5_120, 60.0, 9.0, "high", None, 0.0, True),    # low fs
        make_params("100k", N_100K, 12_800, 50.0, 5.0, "full", 40, 0.05, True),
        make_params("short", round(SHORT_SECONDS * 10_000), 10_000, 60.0, 5.0, "low_odd", None, 0.0, True),
        make_params("short", round(SHORT_SECONDS * 10_000), 10_000, 59.93, 0.5, "full", 30, 0.0, True),
    ]


def draw_params(rng, tier):
    while True:
        fs = 10_000 if tier == "1M" else int(rng.choice(FS_VALUES, p=FS_WEIGHTS))
        n = {"1M": N_1M, "100k": N_100K}.get(tier) or round(SHORT_SECONDS * fs)
        p = make_params(
            tier, n, fs,
            float(rng.choice(F0_VALUES)), float(rng.choice(THD_LEVELS)),
            str(rng.choice(list(ORDER_SETS))),
            SNR_VALUES[int(rng.integers(len(SNR_VALUES)))],
            float(rng.choice(DC_VALUES)), bool(rng.random() < 0.85),
        )
        if usable_orders(p["order_set"], p["f0"], p["fs"]):
            return p


def build_trials(rng, n_100k, n_1m, n_short):
    trials = [(p, int(rng.integers(2 ** 31 - 1))) for p in corner_cases()]
    for tier, count in (("1M", n_1m), ("100k", n_100k), ("short", n_short)):
        trials += [(draw_params(rng, tier), int(rng.integers(2 ** 31 - 1))) for _ in range(count)]
    return trials


# ---------------------------------------------------------------------------
# Summary and gate
# ---------------------------------------------------------------------------
def summarize(df):
    L = []
    ok = df[df["status"] == "ok"]
    L.append("=== Validation summary ===")
    L.append(f"Trials: {len(df)}   ok: {len(ok)}   analyzer/loader failures: "
             f"{int(df['status'].str.contains('error').sum())}   skipped: "
             f"{int(df['status'].str.startswith('skipped').sum())}")

    L.append("\n-- By tier (worst case / mean). gated=True rows count toward the pass/fail gate --")
    L.append(f"{'tier':8s} {'gated':>5s} {'n':>4s} {'THD abs pp':>18s} {'THD rel %':>16s} {'f0 err bins':>14s} "
             f"{'RMS rel %':>16s} {'analyze s (med)':>16s}")
    for (tier, is_gated), g in ok.groupby(["tier", "gated"], sort=False):
        L.append(f"{tier:8s} {str(is_gated):>5s} {len(g):4d} {g.thd_abs_err_pp.max():9.4f}/{g.thd_abs_err_pp.mean():8.4f} "
                 f"{g.thd_rel_err_pct.max():8.3f}/{g.thd_rel_err_pct.mean():7.3f} "
                 f"{g.f0_err_bins.max():7.3f}/{g.f0_err_bins.mean():6.3f} "
                 f"{g.rms_rel_err_pct.max():8.5f}/{g.rms_rel_err_pct.mean():7.5f} {g.analyze_s.median():12.3f}")

    gated = ok[ok["gated"]]
    for col, label in (("thd_true", "true THD (%)"), ("snr_db", "SNR (dB)"), ("f0", "fundamental (Hz)"),
                       ("order_set", "harmonic set")):
        L.append(f"\n-- Gated tiers, by {label}: worst THD abs pp / worst THD rel % --")
        g = gated.assign(**{col: gated[col].fillna("none") if col == "snr_db" else gated[col]})
        if col == "thd_true":
            g = g.assign(thd_true=g["thd_true"].round(1))
        for key, grp in g.groupby(col):
            L.append(f"  {str(key):>10s}  n={len(grp):3d}   {grp.thd_abs_err_pp.max():8.4f} pp   {grp.thd_rel_err_pct.max():8.3f} %")

    if len(ok):
        w = ok.loc[ok["thd_abs_err_pp"].idxmax()]
        L.append("\n-- Worst THD trial overall (rerun it with its seed) --")
        snr_txt = "none" if pd.isna(w.snr_db) else f"{w.snr_db:g} dB"
        L.append(f"  tier={w.tier} seed={int(w.seed)} f0={w.f0} fs={int(w.fs)} n={int(w.n)} set={w.order_set} "
                 f"snr={snr_txt} dc={w.dc} random_phase={w.random_phase}")
        L.append(f"  true THD {w.thd_true:.4f}%  measured {w.thd_meas:.4f}%  error {w.thd_signed_err_pp:+.4f} pp")
    return "\n".join(L)


def check_gate(df, gate_thd, gate_f0, gate_rms):
    """Return a list of failure messages (empty list = gate passed)."""
    g = df[df["gated"] & ~df["status"].str.startswith("skipped")]
    fails = []
    bad = g[g["status"] != "ok"]
    if len(bad):
        fails.append(f"{len(bad)} gated trial(s) failed outright, e.g. seed {int(bad.iloc[0].seed)}: {bad.iloc[0].status}")
    ok = g[g["status"] == "ok"]
    for col, limit, unit in (("thd_abs_err_pp", gate_thd, "pp"), ("f0_err_bins", gate_f0, "bins"),
                             ("rms_rel_err_pct", gate_rms, "%")):
        worst = ok.loc[ok[col].idxmax()]
        if worst[col] > limit:
            fails.append(f"{col} worst {worst[col]:.5f} {unit} > limit {limit} (seed {int(worst.seed)})")
    return fails


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate the analyzer against known synthetic signals.")
    ap.add_argument("--seed", type=int, default=MASTER_SEED, help="master seed (default %(default)s)")
    ap.add_argument("--n-1m", type=int, default=30, help="random trials at 1M samples")
    ap.add_argument("--n-100k", type=int, default=300, help="random trials at 100k samples")
    ap.add_argument("--n-short", type=int, default=100, help="random trials at 0.2 s (not gated)")
    ap.add_argument("--quick", action="store_true", help="small run to smoke-test the script")
    ap.add_argument("--no-gate", action="store_true", help="do not fail on gate limits")
    ap.add_argument("--gate-thd-pp", type=float, default=GATE_THD_PP)
    ap.add_argument("--gate-f0-bins", type=float, default=GATE_F0_BINS)
    ap.add_argument("--gate-rms-pct", type=float, default=GATE_RMS_PCT)
    ap.add_argument("--gate-min-snr", type=float, default=GATE_MIN_SNR_DB,
                    help="noisy trials below this SNR (dB) are reported but not gated")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.quick:
        args.n_1m, args.n_100k, args.n_short = 2, 20, 10

    rng = np.random.default_rng(args.seed)
    trials = build_trials(rng, args.n_100k, args.n_1m, args.n_short)
    print(f"Running {len(trials)} trials + 1 CSV end-to-end case (master seed {args.seed})...")

    rows, t_start = [], time.perf_counter()
    for i, (p, seed) in enumerate(trials, 1):
        rows.append(run_trial(p, seed))
        if i % 25 == 0 or i == len(trials):
            print(f"  {i}/{len(trials)} done ({time.perf_counter() - t_start:.0f} s)")
    rows.append(run_csv_end_to_end(int(rng.integers(2 ** 31 - 1))))
    print(f"  CSV end-to-end done ({time.perf_counter() - t_start:.0f} s total)")

    df = pd.DataFrame(rows)
    # A trial counts toward the gate if it is a long record AND clean enough (noiseless or SNR >= min)
    df["gated"] = df["tier"].isin(GATED_TIERS) & (df["snr_db"].isna() | (df["snr_db"] >= args.gate_min_snr))
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "validation.csv"), index=False)
    text = summarize(df)

    fails = [] if args.no_gate else check_gate(df, args.gate_thd_pp, args.gate_f0_bins, args.gate_rms_pct)
    gate_line = ("Gate: skipped" if args.no_gate else
                 f"Gate (THD<={args.gate_thd_pp} pp, f0<={args.gate_f0_bins} bin, RMS<={args.gate_rms_pct} %): "
                 + ("PASS" if not fails else "FAIL"))
    text += "\n\n" + gate_line + "".join(f"\n  - {f}" for f in fails)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nWrote {os.path.join(args.out_dir, 'validation.csv')} and summary.txt")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())