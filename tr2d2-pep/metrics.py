import pandas as pd
from math import sqrt

def summarize_metrics(skip, csv_path: str, save_path: str | None = None) -> pd.DataFrame:
    """
    Compute mean and standard deviation for all columns except the first
    (assumed non-numeric identifier like 'Peptide Sequence').

    Returns a DataFrame with rows = column names and columns = ['mean','std','count'].
    Uses sample std (ddof=1). Non-numeric cells are coerced to NaN.
    """
    df = pd.read_csv(csv_path)
    vals = df.iloc[:, skip:].apply(pd.to_numeric, errors='coerce')  # columns 2..end
    stats = vals.agg(['mean', 'std', 'count']).T  # shape: (num_metrics, 3)
    if save_path:
        stats.to_csv(save_path, index=True)
    return stats

def summarize_list(xs, ddof = 1):
    # Clean & coerce to float
    vals = []
    for x in xs:
        if x is None or x == "":
            continue
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            continue

    n = len(vals)
    if n == 0:
        raise ValueError("No numeric values found.")
    if n <= ddof:
        raise ValueError(f"Need at least {ddof + 1} numeric values; got {n}.")

    # Welford’s algorithm (one pass, stable)
    mean = 0.0
    M2 = 0.0
    count = 0
    for v in vals:
        count += 1
        delta = v - mean
        mean += delta / count
        M2 += delta * (v - mean)

    var = M2 / (count - ddof)
    std = sqrt(var)

    result = {"mean": mean, "std": std, "count": count}

    return result

def csv_column_to_list(path: str, column: str, *, dropna: bool = True):
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
    s = df[column]
    if dropna:
        s = s.dropna()
    return s.tolist()

def main():
    path = ""
    prot_name = ""
    stats = summarize_metrics(skip=1, csv_path=f"{path}/{prot_name}_generation_results.csv", 
                              save_path=f"{path}/results_summary.csv")

    print(stats)
    
if __name__ == '__main__':
    main()