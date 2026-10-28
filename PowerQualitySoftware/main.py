"""Power Quality Analyzer - entry point.

Loads a time-domain voltage waveform from a CSV, converts it to the frequency domain with an FFT,
and reports RMS voltage, fundamental frequency and total harmonic distortion (THD).

Usage:
    python main.py                          # uses CreatedSyntheticData.csv
    python main.py path/to/data.csv
    python main.py data.csv --time-col t --voltage-col combine --harmonics 50 --no-plot
"""

import argparse
import sys

import numpy as np
import pandas as pd


class AnalyzerError(Exception):
    """A problem with the input data or settings that we can explain to the user."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def load_waveform(path, time_col, voltage_col):
    """Read the CSV and return (t, v) as clean float arrays."""
    try:
        df = pd.read_csv(path, usecols=[time_col, voltage_col])
    except FileNotFoundError:
        raise AnalyzerError(f"File not found: '{path}'. Check the path and current folder.")
    except IsADirectoryError:
        raise AnalyzerError(f"'{path}' is a folder, not a CSV file.")
    except PermissionError:
        raise AnalyzerError(f"No permission to read '{path}'. Is it open in another program?")
    except pd.errors.EmptyDataError:
        raise AnalyzerError(f"'{path}' is empty.")
    except pd.errors.ParserError as e:
        raise AnalyzerError(f"'{path}' is not a valid CSV ({e}).")
    except UnicodeDecodeError:
        raise AnalyzerError(f"'{path}' is not a text CSV file (could not decode it).")
    except ValueError:
        # pandas raises ValueError when usecols names are not in the header
        try:
            found = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            found = []
        raise AnalyzerError(
            f"Could not find column(s) '{time_col}' and '{voltage_col}' in '{path}'. "
            f"Columns found: {found}. Use --time-col / --voltage-col to pick the right ones."
        )
    except MemoryError:
        raise AnalyzerError(f"'{path}' is too large to load into memory.")

    # Anything that is not a number (text, blanks, 'ERROR', ...) becomes NaN, then gets dropped
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[voltage_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(t) & np.isfinite(v)
    n_bad = int(np.count_nonzero(~good))

    if n_bad == len(t):
        raise AnalyzerError("No numeric data found in the selected columns.")
    if n_bad > 0:
        pct = 100 * n_bad / len(t)
        if pct > 5:
            raise AnalyzerError(
                f"{n_bad} of {len(t)} rows ({pct:.1f}%) are missing or non-numeric. "
                "File looks corrupt; refusing to guess."
            )
        # A bad voltage cell with a good timestamp is repaired by linear interpolation, which
        # keeps the time axis evenly spaced. A bad timestamp can't be repaired, so that row is
        # dropped (the uniform-sampling check will then flag the gap).
        t_ok = np.isfinite(t)
        fixable = t_ok & ~np.isfinite(v)
        if np.any(fixable):
            v_ok = np.isfinite(v)
            v = v.copy()
            v[fixable] = np.interp(t[fixable], t[v_ok & t_ok], v[v_ok & t_ok])
        keep = t_ok
        print(f"Warning: {n_bad} missing/non-numeric row(s) ({pct:.2f}%): "
              f"{int(np.count_nonzero(fixable))} voltage value(s) interpolated, "
              f"{int(np.count_nonzero(~keep))} row(s) dropped.")
        t, v = t[keep], v[keep]

    return t, v


def check_sampling(t, min_samples=256, tolerance=0.01):
    """Verify the time axis is usable for an FFT and return (dt, fs)."""
    if len(t) < min_samples:
        raise AnalyzerError(
            f"Only {len(t)} samples; need at least {min_samples} for a meaningful analysis."
        )

    steps = np.diff(t)
    dt = float(np.median(steps))
    if dt <= 0:
        raise AnalyzerError("Time column must be increasing (found a non-positive time step).")

    # The FFT assumes evenly spaced samples
    irregular = np.abs(steps - dt) > tolerance * dt
    if np.any(irregular):
        n_irr = int(np.count_nonzero(irregular))
        raise AnalyzerError(
            f"Sampling is not uniform: {n_irr} time steps differ from {dt:.3g} s by more than "
            f"{tolerance:.0%}. Resample the data to a fixed rate first."
        )

    return dt, 1.0 / dt


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(v, dt, fs, num_harmonics=50):
    """Return a dict of results: RMS, fundamental, THD, per-harmonic magnitudes, spectrum."""
    N = len(v)
    rms = float(np.sqrt(np.mean(v ** 2)))

    ac = v - np.mean(v)  # remove DC so it can't be mistaken for the fundamental
    if np.std(ac) < 1e-12 * max(1.0, abs(np.mean(v))):
        raise AnalyzerError("Signal is constant (no AC component), so there is nothing to analyze.")

    # Hann window: the record is rarely a whole number of cycles, and an unwindowed FFT would
    # make peak heights depend on where each tone falls between bins (scalloping).
    window = np.hanning(N)
    spectrum = np.abs(np.fft.rfft(ac * window))
    freqs = np.fft.rfftfreq(N, dt)
    amplitude = 2 * spectrum / np.sum(window)  # peak amplitude in volts, for plotting

    # Skip the DC bin; fundamental = strongest remaining peak
    k0 = int(np.argmax(spectrum[1:])) + 1
    fundamental_freq = float(freqs[k0])
    bin_res = fs / N

    if fundamental_freq <= 0:
        raise AnalyzerError("Could not detect a fundamental frequency.")
    if fundamental_freq * 2 >= fs / 2:
        raise AnalyzerError(
            f"Fundamental ({fundamental_freq:.1f} Hz) is too close to the Nyquist limit "
            f"({fs / 2:.0f} Hz) to see any harmonics. Sample faster."
        )

    def band_mag(center, k=2):
        # Energy over the tone's main lobe, so the result doesn't depend on bin alignment
        lo = max(center - k, 1)
        return float(np.sqrt(np.sum(spectrum[lo:center + k + 1] ** 2)))

    fund_mag = band_mag(k0)
    if fund_mag <= 0:
        raise AnalyzerError("Fundamental magnitude is zero; cannot compute THD.")

    orders, harm_mags, harm_freqs = [], [], []
    for h in range(2, num_harmonics + 1):
        target = h * fundamental_freq
        if target >= fs / 2:  # beyond Nyquist -> would alias
            break
        # Fundamental is only known to ~one bin and that error scales with h, so search a window
        half_width = (2 + h) * bin_res
        in_window = np.abs(freqs - target) <= half_width
        in_window[0] = False
        idx = int(np.argmax(np.where(in_window, spectrum, -np.inf)))
        orders.append(h)
        harm_mags.append(band_mag(idx))
        harm_freqs.append(float(freqs[idx]))

    used = len(orders)
    if used == 0:
        raise AnalyzerError("No harmonics fit below the Nyquist frequency.")

    harm_mags = np.array(harm_mags)
    thd = 100 * float(np.sqrt(np.sum(harm_mags ** 2))) / fund_mag

    return {
        "N": N,
        "duration": N * dt,
        "fs": fs,
        "rms": rms,
        "dc_offset": float(np.mean(v)),
        "fundamental_freq": fundamental_freq,
        "orders": np.array(orders),
        "harm_pct": 100 * harm_mags / fund_mag,
        "harm_freqs": np.array(harm_freqs),
        "harmonics_used": used,
        "harmonics_requested": num_harmonics,
        "thd": thd,
        "freqs": freqs,
        "amplitude": amplitude,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_report(r, top=5):
    print("\n=== Power Quality Report ===")
    print(f"Samples:               {r['N']:,} ({r['duration']:.2f} s at {r['fs']:.0f} Hz)")
    print(f"RMS voltage:           {r['rms']:.3f} V")
    print(f"DC offset:             {r['dc_offset']:.4f} V")
    print(f"Fundamental frequency: {r['fundamental_freq']:.2f} Hz")
    print(f"Harmonics analysed:    2..{r['orders'][-1]} ({r['harmonics_used']} of "
          f"{r['harmonics_requested'] - 1} requested)")
    print(f"THD:                   {r['thd']:.2f} %")

    order = np.argsort(r["harm_pct"])[::-1][:top]
    print(f"\nLargest harmonics (% of fundamental):")
    for i in order:
        print(f"  H{r['orders'][i]:<3d} {r['harm_freqs'][i]:8.1f} Hz   {r['harm_pct'][i]:6.2f} %")

    if r["harmonics_used"] < r["harmonics_requested"] - 1:
        print(f"\nNote: only {r['harmonics_used']} harmonics fit below Nyquist ({r['fs'] / 2:.0f} Hz).")
    if not (45 <= r["fundamental_freq"] <= 65):
        print(f"Note: fundamental ({r['fundamental_freq']:.2f} Hz) is not near 50/60 Hz. "
              "Check the time column units (seconds?).")


def show_plots(t, v, r, max_freq=1000):
    """Time-domain and spectrum plots. Plot problems never abort the analysis."""
    try:
        import matplotlib.pyplot as plt

        f0 = r["fundamental_freq"]
        dt = t[1] - t[0]
        n_show = int(min(len(t), max(100, 5 / f0 / dt)))  # about 5 cycles

        mask = (r["freqs"] > 0) & (r["freqs"] <= max_freq)

        plt.figure(figsize=(10, 12))
        plt.subplot(2, 1, 1)
        plt.plot(t[:n_show], v[:n_show])
        plt.title("Time-Domain Voltage")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(r["freqs"][mask], r["amplitude"][mask])
        plt.title(f"Frequency Spectrum (THD = {r['thd']:.2f}%)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib is not installed, skipping plots.")
    except Exception as e:  # no display, backend problems, etc.
        print(f"Warning: could not display plots ({type(e).__name__}: {e}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="FFT-based harmonic / THD analyzer for voltage waveforms.")
    p.add_argument("csv", nargs="?", default="CreatedSyntheticData.csv", help="input CSV file")
    p.add_argument("--time-col", default="t", help="time column name (seconds)")
    p.add_argument("--voltage-col", default="combine", help="voltage column name")
    p.add_argument("--harmonics", type=int, default=50, help="highest harmonic order (default 50)")
    p.add_argument("--no-plot", action="store_true", help="skip the plots")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.harmonics < 2:
            raise AnalyzerError("--harmonics must be at least 2.")

        t, v = load_waveform(args.csv, args.time_col, args.voltage_col)
        dt, fs = check_sampling(t)
        results = analyze(v, dt, fs, args.harmonics)
        print_report(results)
        if not args.no_plot:
            show_plots(t, v, results)
        return 0

    except AnalyzerError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except MemoryError:
        print("\nError: ran out of memory. Try a shorter recording.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as e:  # last resort: still no raw traceback for the user
        print(f"\nUnexpected error ({type(e).__name__}): {e}", file=sys.stderr)
        print("This is probably a bug; please report the input file and this message.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())