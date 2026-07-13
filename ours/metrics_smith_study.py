import os
import pickle
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, wasserstein_distance
from sklearn.metrics import r2_score

# ============================================================
# Model directory metrics (predicted vs human_data per .pkl)
# ============================================================

BASE_DIRS = [
    "/nfs/projects/ptgt_predictions/smith_dataset/ours",
    "/nfs/projects/ptgt_predictions/smith_dataset/ours_no_corr",
    "/nfs/projects/ptgt_predictions/smith_dataset/tafasca_baseline",
    "/nfs/projects/ptgt_predictions/smith_dataset/bansal_baseline"
]

def concordance_corr_coeff(x, y):
    """Lin's concordance correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)

def compute_metrics(h, p):
    """Compute similarity / error metrics between two sequences."""
    h = np.asarray(h, dtype=float)
    p = np.asarray(p, dtype=float)

    if h.shape != p.shape:
        raise ValueError(f"Length mismatch: len(h)={len(h)}, len(p)={len(p)}")
    if h.size < 2:
        raise ValueError("Need at least 2 points for these metrics")

    metrics = {}

    # Correlation-type metrics (shape-oriented)
    try:
        r, pval = pearsonr(h, p)
    except Exception:
        r, pval = np.nan, np.nan
    metrics["pearson_r"] = float(r)
    metrics["pearson_p"] = float(pval)

    try:
        rho, p_rho = spearmanr(h, p)
    except Exception:
        rho, p_rho = np.nan, np.nan
    metrics["spearman_rho"] = float(rho)
    metrics["spearman_p"] = float(p_rho)

    try:
        tau, p_tau = kendalltau(h, p)
    except Exception:
        tau, p_tau = np.nan, np.nan
    metrics["kendall_tau"] = float(tau)
    metrics["kendall_p"] = float(p_tau)

    # Cosine similarity (shape)
    denom = (np.linalg.norm(h) * np.linalg.norm(p))
    if denom == 0:
        cos_sim = np.nan
    else:
        cos_sim = float(np.dot(h, p) / denom)
    metrics["cosine_similarity"] = cos_sim

    # Error metrics
    diff = h - p
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    metrics["rmse"] = rmse
    metrics["mae"] = mae

    # Concordance correlation coefficient
    try:
        ccc = float(concordance_corr_coeff(h, p))
    except Exception:
        ccc = np.nan
    metrics["ccc"] = ccc

    # R^2
    try:
        r2 = float(r2_score(h, p))
    except Exception:
        r2 = np.nan
    metrics["r2"] = r2

    # Distribution distance (Earth Mover's / Wasserstein)
    try:
        emd = float(wasserstein_distance(h, p))
    except Exception:
        emd = np.nan
    metrics["emd"] = emd

    return metrics

def collect_results(base_dir):
    """
    Walk base_dir, compute metrics for each .pkl.
    Returns a list of metric dicts.
    """
    all_metrics = []

    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".pkl"):
                continue

            pkl_path = os.path.join(root, fname)
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                human = data["human_data"]
                pred = data["predicted_data"]

                metrics = compute_metrics(human, pred)
                all_metrics.append(metrics)

                # Per-file metrics (comment out if too verbose)
                print(f"\n{pkl_path}:")
                print(f"  Pearson:  r = {metrics['pearson_r']:.4f}, p = {metrics['pearson_p']:.4e}")
                print(f"  Spearman: ρ = {metrics['spearman_rho']:.4f}, p = {metrics['spearman_p']:.4e}")
                print(f"  Kendall:  τ = {metrics['kendall_tau']:.4f}, p = {metrics['kendall_p']:.4e}")
                print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE:  {metrics['mae']:.4f}")
                print(f"  CCC:  {metrics['ccc']:.4f}")
                print(f"  R²:   {metrics['r2']:.4f}")
                print(f"  EMD:  {metrics['emd']:.4f}")

            except Exception as e:
                print(f"Error processing {pkl_path}: {e}")

    return all_metrics

def average_metrics(metric_list):
    """
    Given a list of metric dicts, compute the average for each metric (ignoring NaNs).
    Returns a dict metric_name -> average_value or np.nan if none.
    """
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    avg = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=float)
        if np.all(np.isnan(vals)):
            avg[k] = np.nan
        else:
            avg[k] = float(np.nanmean(vals))
    return avg


# ============================================================
# Human-reference metrics (Smith study version: human vs human)
# ============================================================

def compute_human_reference_c_metrics():
    """
    Human-reference metrics for the C data:
    uses your original fixation_data_next_token_c logic and computes
    shape metrics for all human–human pairs, then averages them.
    """
    fixation_data = pickle.load(open("/nfs/projects/smith_study.pkl", "rb"))
    holdoutset_list = ["p2","p4","p6","p7","p8","p9","p10","p11","p13","p14",
                       "p15","p16","p17","p18","p20","p21"]

    human_metric_list = []

    for holdout in holdoutset_list[:]:
        temp_score = []
        for method_id in list(fixation_data.keys())[:]:
            # your original structure: allids over all keys
            allids = list(fixation_data.keys())

            # find holdoutset for this holdout
            for id_ in allids:
                participant = id_.split("_")[0]
                if participant == holdout:
                    if id_ in fixation_data and fixation_data[id_] != []:
                        holdoutset = fixation_data[id_]
                        break

            # compare against all non-holdouts
            for id_ in allids:
                participant = id_.split("_")[0]
                if participant == holdout:
                    continue
                if id_ in fixation_data and fixation_data[id_] != []:
                    nonholdout = fixation_data[id_]
                else:
                    continue

                holdoutdurationlist = []
                holdoutdurationdict = {}
                for d in holdoutset:
                    token = d["token"]
                    duration = d["duration"]
                    fuunction = d["function"]
                    holdoutdurationdict[(token, fuunction)] = duration

                if nonholdout == []:
                    continue

                ptgtlist = []
                ptgtdict = {}
                for d in nonholdout:
                    token = d["token"]
                    duration = d["duration"]
                    function = d["function"]
                    ptgtdict[(token, function)] = duration

                for (token, function) in holdoutdurationdict:
                    if (token, function) in ptgtdict:
                        duration_non = ptgtdict[(token, function)]
                        duration_hold = holdoutdurationdict[(token, function)]
                        ptgtlist.append(duration_non)
                        holdoutdurationlist.append(duration_hold)

                if len(ptgtlist) <= 1:
                    continue

                # here we replace the original pearsonr() call with full shape metrics
                metrics = compute_metrics(holdoutdurationlist, ptgtlist)
                human_metric_list.append(metrics)
                temp_score.append(metrics["pearson_r"])

        if temp_score:
            avg_corr = sum(temp_score) / len(temp_score)
        else:
            avg_corr = float("nan")
        print(f"C human-reference Pearson for {holdout}: {avg_corr:.4f}")

    if human_metric_list:
        return average_metrics(human_metric_list)
    else:
        return None


# ============================================================
# Main: compute model metrics, human-reference metrics, print table
# ============================================================

def main():
    dir_summaries = {}

    # 1) Model directory metrics
    for base in BASE_DIRS:
        print(f"\n==============================")
        print(f"Processing directory: {base}")
        print(f"==============================")

        metrics_list = collect_results(base)
        if not metrics_list:
            dir_summaries[base] = None
            print(f"No valid .pkl files found in {base}")
            continue

        avg = average_metrics(metrics_list)
        dir_summaries[base] = avg

        print(f"\nAverage metrics for {base}:")
        print(f"  Pearson r:          {avg['pearson_r']:.4f}")
        print(f"  Spearman ρ:         {avg['spearman_rho']:.4f}")
        print(f"  Kendall τ:          {avg['kendall_tau']:.4f}")
        print(f"  Cosine similarity:  {avg['cosine_similarity']:.4f}")
        print(f"  RMSE:               {avg['rmse']:.4f}")
        print(f"  MAE:                {avg['mae']:.4f}")
        print(f"  CCC:                {avg['ccc']:.4f}")
        print(f"  R²:                 {avg['r2']:.4f}")
        print(f"  EMD:                {avg['emd']:.4f}")


    print("\n==============================")
    print("Computing Smith study human-reference (human vs human) metrics")
    print("==============================")
    human_c_avg = compute_human_reference_c_metrics()

    
    # 3) Pretty-printed, aligned shape-metrics table with custom ordering
    print("\n========== Shape Metrics Table (Pretty Print) ==========")

    # Desired row order
    desired_order = [
        "human_reference",
        "bansal_baseline",
        "tafasca_baseline",
        "ours_no_corr",
        "ours"
    ]

    # Build rows
    rows = []
    for name in desired_order:
        if name == "":
            rows.append(["", "", "", "", ""])  # blank separator row
            continue

        if name == "human_reference":
            if human_c_avg is None:
                rows.append([name, "n/a","n/a",  "n/a", "n/a"])
            else:
                rows.append([
                    name,
                    f"{human_c_avg['pearson_r']:.4f}",
                    f"{human_c_avg['spearman_rho']:.4f}",
                    f"{human_c_avg['kendall_tau']:.4f}",
                    f"{human_c_avg['cosine_similarity']:.4f}",
                ])
            continue
        
        # Model directories: match by basename
        matching_key = None
        for base in dir_summaries:
            if os.path.basename(base) == name:
                matching_key = base
                break

        if matching_key is None:
            rows.append([name, "n/a", "n/a", "n/a", "n/a"])
            continue

        avg = dir_summaries[matching_key]
        if avg is None:
            rows.append([name, "n/a", "n/a", "n/a", "n/a"])
        else:
            rows.append([
                name,
                f"{avg['pearson_r']:.4f}",
                f"{avg['spearman_rho']:.4f}",
                f"{avg['kendall_tau']:.4f}",
                f"{avg['cosine_similarity']:.4f}",
            ])

    headers = ["Directory", "Pearson r", "Spearman ρ", "Kendall τ", "Cosine sim"]

    # Compute column widths
    col_widths = [
        max(len(str(row[i])) for row in ([headers] + rows))
        for i in range(len(headers))
    ]

    def fmt_row(row):
        # Left-align directory column, right-align numeric columns
        return "  ".join(
            (str(row[i]).ljust(col_widths[i]) if i == 0 else str(row[i]).rjust(col_widths[i]))
            for i in range(len(row))
        )

    # Print header
    print(fmt_row(headers))
    print("  ".join("-" * col_widths[i] for i in range(len(headers))))

    # Print rows
    for row in rows:
        print(fmt_row(row))

if __name__ == "__main__":
    main()


