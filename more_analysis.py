# bayes_mode_full_analysis.py
"""
Full mode-separated Bayesian posterior analysis and predictive uncertainty propagation.

- Requires:
    * multi_stations_input.read_input()
    * forward_model_multi_station.forward_model_multi_station(...)
    * multi_station_sampled_params.npy in same folder (PyDREAM output)

- Behavior:
    * Loads posterior samples, removes burn-in (20% default)
    * Fits a GMM to detect modes (user-selectable n_modes)
    * For each mode:
        - computes mean, median, MAP (KDE-based per-dim), covariance, weight
        - saves corner plot
        - propagates mode-specific posterior samples through forward model to compute
          predictive ensemble and credible intervals (2.5%/50%/97.5%) per station & component
        - saves predictive plots: MAP line, posterior-mean line, median, and shaded 95% band
    * Also produces predictive plot using the FULL posterior (no clustering)
    * NO observational noise/sigma is injected into the predictive bands — they are purely from parameter uncertainty.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import seaborn as sns

# Local imports (your forward model + input)
import multi_stations_input as input_parameters
from forward_model_multi_station import forward_model_multi_station

# -------------------------
# Settings
# -------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODE_OUT_DIR = os.path.join(SCRIPT_DIR, "mode_separation")
os.makedirs(MODE_OUT_DIR, exist_ok=True)

# User-adjustable parameters
BURN_IN_FRAC = 0.2          # fraction of each chain to discard as burn-in
N_MODES = 3                 # number of GMM components to fit (you can change)
N_PRED_SAMPLES = 400        # how many posterior draws to propagate for predictive bands
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------
# Utilities
# -------------------------
def load_pydream_flat(samples_file="multi_station_sampled_params.npy", burn_in_frac=BURN_IN_FRAC):
    """Load PyDREAM samples and flatten chains removing burn-in. Returns post (N x D)."""
    path = os.path.join(SCRIPT_DIR, samples_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Posterior file not found: {path}")
    arr = np.load(path, allow_pickle=True)
    # Normalize to list-of-chains
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        chains = list(arr)
    elif isinstance(arr, np.ndarray) and arr.ndim == 3:
        # assume shape (nchains, niters, nparams)
        chains = [arr[i] for i in range(arr.shape[0])]
    elif isinstance(arr, np.ndarray) and arr.ndim == 2:
        chains = [arr]
    else:
        raise RuntimeError(f"Unsupported posterior format: shape {getattr(arr, 'shape', None)}")
    burn_in = int(chains[0].shape[0] * burn_in_frac)
    post = np.vstack([ch[burn_in:] for ch in chains])
    print(f"Loaded posterior. Chains: {len(chains)}. Post shape after burn-in: {post.shape}")
    return post, chains

def safe_param_extract(draw, param_names, defaults_first4=True):
    """Given a draw (1D) and param_names, extract a,b,c,E robustly."""
    try:
        a = float(draw[param_names.index('a')])
        b = float(draw[param_names.index('b')])
        c = float(draw[param_names.index('c')])
        E = float(draw[param_names.index('E')])
        return a, b, c, E
    except Exception:
        if defaults_first4:
            a, b, c, E = [float(x) for x in draw[:4]]
            return a, b, c, E
        raise

def kde_map_1d(samples_1d):
    """Return MAP estimate for a 1D array using gaussian_kde."""
    try:
        kde = gaussian_kde(samples_1d)
        xs = np.linspace(np.min(samples_1d), np.max(samples_1d), 800)
        vals = kde(xs)
        return xs[np.argmax(vals)]
    except Exception:
        # fallback: median
        return np.median(samples_1d)

# -------------------------
# Mode stats & corner plot
# -------------------------
def compute_modes(post, n_modes=N_MODES, random_state=RANDOM_SEED):
    """Fit GMM to post and return labels and the gmm object."""
    gmm = GaussianMixture(n_components=n_modes, covariance_type='full', random_state=random_state)
    labels = gmm.fit_predict(post)
    print(f"GMM fitted: found {len(np.unique(labels))} unique labels.")
    return gmm, labels

def mode_statistics(post, labels, param_names=None, loglike=None, true_params=None, out_dir=MODE_OUT_DIR):
    """Compute stats for each mode and produce corner plots."""
    n_modes = len(np.unique(labels))
    N = post.shape[0]
    modes = []
    for k in sorted(np.unique(labels)):
        idx = np.where(labels == k)[0]
        subset = post[idx]
        weight = len(idx) / N
        mean_k = subset.mean(axis=0)
        median_k = np.median(subset, axis=0)
        cov_k = np.cov(subset.T)

        if loglike is not None:
            # if loglike is per-sample aligned: pick argmax within idx
            try:
                max_ll_idx = idx[np.argmax(loglike[idx])]
                MAP_k = post[max_ll_idx]
            except Exception:
                MAP_k = np.array([kde_map_1d(subset[:, i]) for i in range(subset.shape[1])])
        else:
            MAP_k = np.array([kde_map_1d(subset[:, i]) for i in range(subset.shape[1])])

        dist = np.linalg.norm(mean_k - true_params) if true_params is not None else None

        mode_info = {
            "mode": int(k),
            "indices": idx,
            "count": int(len(idx)),
            "weight": float(weight),
            "mean": mean_k,
            "median": median_k,
            "MAP": MAP_k,
            "cov": cov_k,
            "distance_to_true": float(dist) if dist is not None else None,
            "samples": subset
        }
        modes.append(mode_info)

        # corner plot (simple pairwise scatter + KDE on diagonal)
        corner_path = os.path.join(out_dir, f"corner_mode_{k}.png")
        try:
            save_corner_plot(subset, param_names, title=f"Mode {k} (weight={weight:.3f})", out_path=corner_path)
            print(f"Saved corner: {corner_path}")
        except Exception as e:
            print("Corner plot failed for mode", k, e)
    # write text summary
    summary_path = os.path.join(out_dir, "mode_summary.txt")
    with open(summary_path, "w") as f:
        for m in modes:
            f.write(f"Mode {m['mode']}:\n")
            f.write(f"  weight = {m['weight']:.4f}\n")
            f.write(f"  count  = {m['count']}\n")
            f.write(f"  mean   = {m['mean']}\n")
            f.write(f"  median = {m['median']}\n")
            f.write(f"  MAP    = {m['MAP']}\n")
            if m['distance_to_true'] is not None:
                f.write(f"  dist_to_true = {m['distance_to_true']:.6g}\n")
            f.write("  covariance:\n")
            f.write(str(m['cov']))
            f.write("\n\n")
    print(f"Mode summary saved: {summary_path}")
    return modes

def save_corner_plot(samples, param_names, title="", out_path=None):
    """Minimal corner-like plot (no external dependency)."""
    D = samples.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(3.2*D, 3.2*D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                xs = np.linspace(samples[:, i].min(), samples[:, i].max(), 300)
                try:
                    kde = gaussian_kde(samples[:, i])
                    ax.plot(xs, kde(xs), color='black')
                except Exception:
                    ax.hist(samples[:, i], bins=30, density=True, color='gray', alpha=0.7)
                ax.set_yticks([])
            else:
                ax.scatter(samples[:, j], samples[:, i], s=4, alpha=0.3, color='black')
            if i == D - 1:
                ax.set_xlabel(param_names[j])
            if j == 0:
                ax.set_ylabel(param_names[i])
    plt.suptitle(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=250)
    plt.close(fig)
    gc.collect()

# -------------------------
# Predictive propagation (parameter-driven only)
# -------------------------
def predictive_ensemble_from_samples(samples, param_names, n_draws=N_PRED_SAMPLES):
    """
    Given samples (N x D), draw up to n_draws parameter vectors (without replacement)
    and run forward_model_multi_station for each sample, returning ensemble shaped
    (n_draws, n_comp, nt) aggregated across stations inside the forward model result.
    """
    params_input = input_parameters.read_input()
    time = np.array(params_input['time'])
    Ns = len(params_input['x_prime'])
    COMPONENT_COLS = []
    base_comps = [
        "Epsilon_XX_nanostrain",
        "Epsilon_YY_nanostrain",
        "Epsilon_ZZ_nanostrain",
        "Epsilon_XY_nanostrain",
        "Epsilon_XZ_nanostrain",
        "Epsilon_YZ_nanostrain",
    ]
    for s in range(Ns):
        for comp in base_comps:
            COMPONENT_COLS.append(f"{comp}_FS{s+1:02d}")

    Ntot = samples.shape[0]
    n_draws = min(n_draws, Ntot)
    rng = np.random.default_rng(RANDOM_SEED)
    idxs = rng.choice(Ntot, n_draws, replace=False)

    ensemble_list = []  # will hold arrays shaped (n_comp, nt) per draw
    for sidx in idxs:
        draw = samples[sidx]
        # robust parameter extraction
        try:
            a, b, c, E = safe_param_extract(draw, param_names)
        except Exception:
            # fallback
            a, b, c, E = [float(x) for x in draw[:4]]
        # call forward model: must return a DataFrame with COMPONENT_COLS as columns
        df = forward_model_multi_station(
            params_input['pmax'], params_input['tpeak'], params_input['d'], time,
            np.array(params_input['x_prime']), np.array(params_input['y_prime']),
            params_input['x0_prime'], params_input['y0_prime'],
            params_input['z'], a, b, c, params_input['nu'], params_input['h'], E,
            params_input['theta_deg'], params_input.get('alpha', None)
        )
        # ensure columns exist
        missing = [c for c in COMPONENT_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Forward model output missing columns: {missing}")
        arr = df[COMPONENT_COLS].values.T  # shape (n_comp, nt)
        ensemble_list.append(arr)

    ensemble = np.stack(ensemble_list, axis=0)  # (n_draws, n_comp, nt)
    return ensemble, COMPONENT_COLS, time

def plot_predictive_for_ensemble(ensemble, COMPONENT_COLS, time, out_prefix, mode_tag=None, obs_df=None):
    """
    ensemble: (n_draws, n_comp, nt)
    COMPONENT_COLS: list of component column names length n_comp
    time: 1D array
    """
    n_draws, n_comp, nt = ensemble.shape
    # quantiles across draws
    lower = np.percentile(ensemble, 2.5, axis=0)
    median = np.percentile(ensemble, 50, axis=0)
    upper = np.percentile(ensemble, 97.5, axis=0)
    mean_pred = np.mean(ensemble, axis=0)

    # For each station (6 components per station)
    n_per_station = 6
    Ns = int(len(COMPONENT_COLS) / n_per_station)

    for s in range(Ns):
        comp_slice = slice(s * n_per_station, (s+1) * n_per_station)
        comps = COMPONENT_COLS[comp_slice]
        short_labels = [c.replace('_nanostrain', '') for c in comps]

        fig, ax = plt.subplots(figsize=(14, 7))
        colors = sns.color_palette("tab10", n_per_station)
        for i in range(n_per_station):
            ax.plot(time, mean_pred[comp_slice.start + i - comp_slice.start], color=colors[i], lw=1.8,
                    label=f"{short_labels[i]} (mean)")
            ax.fill_between(time, lower[comp_slice.start + i - comp_slice.start], upper[comp_slice.start + i - comp_slice.start],
                            color=colors[i], alpha=0.25)
            # optional: plot median or MAP line
            ax.plot(time, median[comp_slice.start + i - comp_slice.start], linestyle='--', lw=1.0, color=colors[i], alpha=0.9)

            # plot observed if provided
            if obs_df is not None and comps[i] in obs_df.columns:
                ax.scatter(time, obs_df[comps[i]].values, color=colors[i], s=12, alpha=0.6, zorder=5)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Strain (nε)")
        title = f"Station {s+1} — Predictive mean & 95% CI"
        if mode_tag:
            title += f" ({mode_tag})"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        out_png = os.path.join(MODE_OUT_DIR, f"{out_prefix}_station_{s+1}" + (f"_{mode_tag}" if mode_tag else "") + ".png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved predictive plot: {out_png}")
        gc.collect()

    return lower, median, upper, mean_pred

# -------------------------
# Parameter uncertainty summary plot (per-parameter bands)
# -------------------------
def plot_parameter_uncertainty(post, param_names):
    D = post.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.2 * D))
    if D == 1:
        axes = [axes]
    for i in range(D):
        s = post[:, i]
        xs = np.linspace(np.min(s), np.max(s), 400)
        try:
            kde = gaussian_kde(s)
            axes[i].plot(xs, kde(xs), color='black', lw=1.6)
        except Exception:
            axes[i].hist(s, bins=40, density=True, alpha=0.5)
        mean = np.mean(s)
        median = np.median(s)
        p05, p16, p84, p95 = np.percentile(s, [5, 16, 84, 95])
        axes[i].axvspan(p05, p95, alpha=0.15, color='gray', label='90% CI')
        axes[i].axvspan(p16, p84, alpha=0.25, color='lightblue', label='1-sigma')
        axes[i].axvline(mean, color='red', lw=1.6, label='mean')
        axes[i].axvline(median, color='blue', lw=1.2, label='median')
        axes[i].set_ylabel(param_names[i])
        axes[i].legend()
    plt.tight_layout()
    out_path = os.path.join(MODE_OUT_DIR, "parameter_uncertainty_summary.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved parameter uncertainty summary: {out_path}")

# -------------------------
# Main workflow
# -------------------------
def main():
    # Load posterior
    post, chains = load_pydream_flat()  # uses default filename in script dir
    # optional: load loglike if you have it named multi_station_logps.npy
    loglike_path = os.path.join(SCRIPT_DIR, "multi_station_logps.npy")
    loglike = None
    if os.path.exists(loglike_path):
        try:
            loglike = np.load(loglike_path, allow_pickle=True)
            if isinstance(loglike, np.ndarray) and loglike.dtype == object:
                # flatten per-chain then remove the burn-in length consistent with chains
                burn_in = int(chains[0].shape[0] * BURN_IN_FRAC)
                loglike = np.hstack([lc[burn_in:] for lc in loglike])
        except Exception:
            loglike = None

    # Param names infer
    D = post.shape[1]
    if D == 5:
        param_names = ['a', 'b', 'c', 'E', 'sigma']
    else:
        param_names = [f"p{i}" for i in range(D)]
    print("Parameter names:", param_names)

    # Fit modes
    gmm, labels = compute_modes(post, n_modes=N_MODES)
    modes = mode_statistics(post, labels, param_names=param_names, loglike=loglike, true_params=None, out_dir=MODE_OUT_DIR)

    # Save parameter summary plot (full posterior)
    plot_parameter_uncertainty(post, param_names)

    # Observed dataset (if exists) for overlay
    obs_csv = os.path.join(SCRIPT_DIR, 'strain_dataset_output_multi.csv')
    obs_df = pd.read_csv(obs_csv) if os.path.exists(obs_csv) else None
    if obs_df is not None:
        print("Loaded observed dataset for overlay:", obs_csv)

    # Predictive plot using FULL posterior
    ensemble_full, COMPONENT_COLS, time = predictive_ensemble_from_samples(post, param_names, n_draws=N_PRED_SAMPLES)
    plot_predictive_for_ensemble(ensemble_full, COMPONENT_COLS, time, out_prefix="predictive_full", mode_tag="full", obs_df=obs_df)

    # Mode-wise predictive and summary
    for m in modes:
        samp = m['samples']
        mode_tag = f"mode{m['mode']}"
        # predictive ensemble for this mode
        if samp.shape[0] < 3:
            print(f"Mode {m['mode']} has too few samples ({samp.shape[0]}). Skipping predictive propagation.")
            continue
        ensemble_mode, COMPONENT_COLS_mode, time_mode = predictive_ensemble_from_samples(samp, param_names, n_draws=N_PRED_SAMPLES)
        plot_predictive_for_ensemble(ensemble_mode, COMPONENT_COLS_mode, time_mode,
                                     out_prefix=f"predictive_{mode_tag}", mode_tag=mode_tag, obs_df=obs_df)

    # Save CSV summary of modes
    import csv
    csv_path = os.path.join(MODE_OUT_DIR, "mode_table.csv")
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        header = ['mode', 'weight', 'count'] + [f"mean_{pn}" for pn in param_names] + [f"map_{pn}" for pn in param_names]
        writer.writerow(header)
        for m in modes:
            row = [m['mode'], m['weight'], m['count']] + list(m['mean']) + list(m['MAP'])
            writer.writerow(row)
    print("Saved mode table:", csv_path)

    print("All done. Outputs in:", MODE_OUT_DIR)


if __name__ == "__main__":
    main()
