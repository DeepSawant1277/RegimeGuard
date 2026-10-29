import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["spread", "imbalance", "depth_total", "mid_return", "rolling_volatility"]

def make_demo_data(n=4000, seed=17):
    """Create reproducible synthetic LOB-like snapshots with an abrupt regime shift."""
    rng = np.random.default_rng(seed)
    shift = int(n * 0.58)
    mid = np.zeros(n)
    mid[0] = 100.0
    bid_size = np.zeros(n)
    ask_size = np.zeros(n)
    bid_price = np.zeros(n)
    ask_price = np.zeros(n)
    for i in range(n):
        regime = i >= shift
        spread = rng.choice([0.01, 0.02, 0.03], p=([.75,.20,.05] if not regime else [.25,.45,.30]))
        imbalance = rng.normal(0.05 if not regime else -0.12, 0.25 if not regime else 0.42)
        bid_size[i] = max(1, 100 + 35 * imbalance + rng.normal(0, 18))
        ask_size[i] = max(1, 100 - 35 * imbalance + rng.normal(0, 18))
        bid_price[i] = mid[i] - spread / 2
        ask_price[i] = mid[i] + spread / 2
        if i < n - 1:
            vol = 0.008 if not regime else 0.025
            drift = 0.0005 * imbalance if not regime else 0.0015 * imbalance
            mid[i+1] = mid[i] + drift + rng.normal(0, vol)
    return pd.DataFrame({"timestamp": np.arange(n), "bid_price": bid_price,
                         "ask_price": ask_price, "bid_size": bid_size, "ask_size": ask_size})

def prepare_data(raw, horizon=10):
    required = {"bid_price", "ask_price", "bid_size", "ask_size"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    df = raw.copy().reset_index(drop=True)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    if len(df) < 500:
        raise ValueError("Please provide at least 500 valid order-book rows.")
    if (df["ask_price"] < df["bid_price"]).any():
        raise ValueError("Some rows have ask_price below bid_price; check the input data.")
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    denom = (df["bid_size"] + df["ask_size"]).replace(0, np.nan)
    df["imbalance"] = (df["bid_size"] - df["ask_size"]) / denom
    df["depth_total"] = df["bid_size"] + df["ask_size"]
    df["mid_return"] = df["mid_price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df["rolling_volatility"] = df["mid_return"].rolling(30, min_periods=2).std().fillna(0)
    df["target"] = (df["mid_price"].shift(-horizon) > df["mid_price"]).astype(int)
    df = df.iloc[:-horizon].copy()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)

def make_model(seed=17):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=60, max_depth=8,
                                         min_samples_leaf=5, class_weight="balanced",
                                         random_state=seed, n_jobs=1))
    ])

def run_experiment(raw, horizon=10, reference_size=400, current_size=150,
                   alpha=0.01, retrain_size=500):
    df = prepare_data(raw, horizon=horizon)
    n = len(df)
    split = int(n * 0.60)
    if split < max(reference_size + current_size, 300):
        raise ValueError("Not enough rows for the selected windows. Increase the dataset size.")
    X, y = df[FEATURES], df["target"]
    model = make_model()
    model.fit(X.iloc[:split], y.iloc[:split])
    static_model = model
    rows = []
    event_rows = []
    recent_alerts = []
    # Evaluate sequentially: only labels already available are eligible for retraining.
    adaptive_model = make_model()
    adaptive_model.fit(X.iloc[:split], y.iloc[:split])
    last_retrain_at = split
    start = max(split, reference_size + current_size)
    # Process a bounded initial reference and rolling windows without using future rows.
    for i in range(start, n):
        shift_detected = False
        pvals = []
        changed_features = []
        ref_start = max(0, i - current_size - reference_size)
        ref_end = i - current_size
        cur_start = i - current_size
        cur_end = i
        if ref_end > ref_start and cur_end > cur_start:
            for feature in FEATURES:
                a = X[feature].iloc[ref_start:ref_end].dropna().to_numpy()
                b = X[feature].iloc[cur_start:cur_end].dropna().to_numpy()
                if len(a) >= 20 and len(b) >= 20:
                    p = float(ks_2samp(a, b, alternative="two-sided", method="auto").pvalue)
                else:
                    p = 1.0
                pvals.append(p)
                if p < alpha / len(FEATURES):
                    changed_features.append(feature)
        # Require at least one feature to remain significant after a simple Bonferroni correction.
        shift_detected = bool(changed_features)
        if shift_detected:
            recent_alerts.append(i)
            event_rows.append({"row": i, "timestamp": df.iloc[i].get("timestamp", i),
                               "features": ", ".join(changed_features),
                               "min_pvalue": min(pvals) if pvals else 1.0})
            # Fit only on rows whose future-horizon target would be known by time i.
            eligible_end = max(0, i - horizon)
            train_start = max(0, eligible_end - retrain_size)
            if eligible_end - train_start >= 100 and i - last_retrain_at >= max(100, current_size):
                adaptive_model = make_model()
                adaptive_model.fit(X.iloc[train_start:eligible_end], y.iloc[train_start:eligible_end])
                last_retrain_at = i
        xrow = X.iloc[[i]]
        t0 = time.perf_counter()
        static_pred = int(static_model.predict(xrow)[0])
        adaptive_pred = int(adaptive_model.predict(xrow)[0])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "row": i, "timestamp": df.iloc[i].get("timestamp", i),
            "mid_price": float(df.iloc[i]["mid_price"]),
            "target": int(y.iloc[i]), "static_pred": static_pred,
            "adaptive_pred": adaptive_pred,
            "static_correct": int(static_pred == int(y.iloc[i])),
            "adaptive_correct": int(adaptive_pred == int(y.iloc[i])),
            "shift_detected": shift_detected,
            "ks_min_pvalue": min(pvals) if pvals else 1.0,
            "inference_ms_both_models": elapsed_ms
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No test rows were evaluated. Increase the dataset size.")
    metrics = {
        "static_accuracy": accuracy_score(out.target, out.static_pred),
        "adaptive_accuracy": accuracy_score(out.target, out.adaptive_pred),
        "static_balanced_accuracy": balanced_accuracy_score(out.target, out.static_pred),
        "adaptive_balanced_accuracy": balanced_accuracy_score(out.target, out.adaptive_pred),
        "static_f1": f1_score(out.target, out.static_pred, zero_division=0),
        "adaptive_f1": f1_score(out.target, out.adaptive_pred, zero_division=0),
        "median_inference_ms": out["inference_ms_both_models"].median()
    }
    return {"predictions": out, "events": pd.DataFrame(event_rows), "metrics": metrics}
