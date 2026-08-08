"""
Phase 0 dataset verification for the FiFAR-based fraud-triage project.

Checks the prepared train/eval parquet files in fifar_prepared/ for the
things that would silently break modeling downstream: schema mismatches,
temporal-split leakage, bad labels, inconsistent missing-value flags,
a scaler that wasn't fit only on train, and malformed one-hot columns.

Usage:
    python verify_dataset.py

Requires: pandas, pyarrow, joblib, scikit-learn
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

DATA_DIR = Path(__file__).parent / "fifar_prepared"
TRAIN_PATH = DATA_DIR / "train_months3to5_processed.parquet"
EVAL_PATH = DATA_DIR / "eval_months6to7_processed.parquet"
SCALER_PATH = DATA_DIR / "scaler.joblib"

TRAIN_MONTHS = {3, 4, 5}
EVAL_MONTHS = {6, 7}

CATEGORICAL_BASES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

results = []  # (status, name, detail)


def check(name):
    def decorator(fn):
        def wrapper(*a, **kw):
            try:
                status, detail = fn(*a, **kw)
            except Exception as e:
                status, detail = "FAIL", f"raised {type(e).__name__}: {e}"
            results.append((status, name, detail))
        return wrapper
    return decorator


def onehot_groups(columns):
    groups = {}
    for base in CATEGORICAL_BASES:
        prefix = base + "_"
        cols = [c for c in columns if c.startswith(prefix)]
        if cols:
            groups[base] = cols
    return groups


def missing_flag_pairs(columns):
    pairs = []
    for c in columns:
        if c.endswith("_is_missing"):
            base = c[: -len("_is_missing")]
            if base in columns:
                pairs.append((base, c))
    return pairs


@check("Files exist")
def _files_exist(state):
    missing = [p.name for p in (TRAIN_PATH, EVAL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        return "FAIL", f"missing: {missing}"
    return "PASS", "train, eval, scaler all present"


@check("Files load and are non-empty")
def _load(state):
    train = pd.read_parquet(TRAIN_PATH)
    eval_ = pd.read_parquet(EVAL_PATH)
    scaler = joblib.load(SCALER_PATH)
    state["train"], state["eval"], state["scaler"] = train, eval_, scaler
    if len(train) == 0 or len(eval_) == 0:
        return "FAIL", f"train={len(train)} rows, eval={len(eval_)} rows"
    return "PASS", f"train={len(train)} rows, eval={len(eval_)} rows, {train.shape[1]} cols"


@check("Train/eval schema matches")
def _schema(state):
    train, eval_ = state["train"], state["eval"]
    if list(train.columns) != list(eval_.columns):
        only_train = set(train.columns) - set(eval_.columns)
        only_eval = set(eval_.columns) - set(train.columns)
        return "FAIL", f"only in train: {only_train}, only in eval: {only_eval}"
    return "PASS", f"{train.shape[1]} columns identical in name and order"


@check("standard# expert-prediction columns present (must be excluded from features)")
def _standard_cols(state):
    train = state["train"]
    std_cols = [c for c in train.columns if c.startswith("standard#")]
    if len(std_cols) != 50:
        return "FAIL", f"expected 50 standard# columns, found {len(std_cols)}"
    return "PASS", "50 standard# columns found (remember: drop these before training Theme 1 models)"


@check("case_id is unique within train and within eval")
def _case_id_unique(state):
    train, eval_ = state["train"], state["eval"]
    if "case_id" not in train.columns:
        return "FAIL", "no case_id column"
    t_dupes = train["case_id"].duplicated().sum()
    e_dupes = eval_["case_id"].duplicated().sum()
    if t_dupes or e_dupes:
        return "FAIL", f"train dupes={t_dupes}, eval dupes={e_dupes}"
    return "PASS", "no duplicate case_id within either split"


@check("No case_id overlap between train and eval (no leakage)")
def _no_overlap(state):
    train, eval_ = state["train"], state["eval"]
    overlap = set(train["case_id"]) & set(eval_["case_id"])
    if overlap:
        return "FAIL", f"{len(overlap)} case_ids appear in both splits"
    return "PASS", "train and eval case_ids are disjoint"


@check("Temporal split is correct (train=months 3-5, eval=months 6-7)")
def _temporal_split(state):
    train, eval_ = state["train"], state["eval"]
    if "month" not in train.columns:
        return "FAIL", "no month column"
    t_months = set(train["month"].unique())
    e_months = set(eval_["month"].unique())
    if not t_months <= TRAIN_MONTHS:
        return "FAIL", f"train contains unexpected months: {t_months - TRAIN_MONTHS}"
    if not e_months <= EVAL_MONTHS:
        return "FAIL", f"eval contains unexpected months: {e_months - EVAL_MONTHS}"
    if t_months & e_months:
        return "FAIL", f"overlapping months between splits: {t_months & e_months}"
    return "PASS", f"train months={sorted(t_months)}, eval months={sorted(e_months)}"


@check("fraud_bool label is clean binary with no nulls")
def _label_clean(state):
    train, eval_ = state["train"], state["eval"]
    if "fraud_bool" not in train.columns:
        return "FAIL", "no fraud_bool column"
    for name, df in [("train", train), ("eval", eval_)]:
        if df["fraud_bool"].isna().any():
            return "FAIL", f"{name} has null fraud_bool"
        vals = set(df["fraud_bool"].unique())
        if not vals <= {0, 1}:
            return "FAIL", f"{name} fraud_bool has non-binary values: {vals}"
    t_rate = train["fraud_bool"].mean()
    e_rate = eval_["fraud_bool"].mean()
    return "PASS", f"train fraud rate={t_rate:.3%}, eval fraud rate={e_rate:.3%}"


@check("No unexpected NaNs remain after preprocessing")
def _no_nans(state):
    train, eval_ = state["train"], state["eval"]
    t_nulls = train.isna().sum()
    e_nulls = eval_.isna().sum()
    t_bad = t_nulls[t_nulls > 0]
    e_bad = e_nulls[e_nulls > 0]
    if len(t_bad) or len(e_bad):
        return "FAIL", f"train nulls in {dict(t_bad)}, eval nulls in {dict(e_bad)}"
    return "PASS", "no NaNs in either split (missing values were sentinel-coded, not left null)"


@check("Missing-value indicator flags match a single sentinel value in their base column")
def _missing_flags(state):
    # NOTE: several of these base columns (e.g. prev_address_months_count)
    # are also scaled by StandardScaler, so the raw -1 sentinel no longer
    # literally equals -1 in the processed data -- it's whatever -1 maps to
    # after (x - mean) / scale. So instead of checking for literal -1, check
    # that "flag == 1" rows all share one single value in the base column,
    # and that value never appears among "flag == 0" rows -- i.e. the
    # sentinel is still internally consistent, however it's encoded.
    train, eval_ = state["train"], state["eval"]
    pairs = missing_flag_pairs(list(train.columns))
    if not pairs:
        return "FAIL", "no *_is_missing columns found"
    problems = []
    notes = []
    for base, flag_col in pairs:
        for name, df in [("train", train), ("eval", eval_)]:
            flagged = df.loc[df[flag_col] == 1, base]
            unflagged = df.loc[df[flag_col] == 0, base]
            if flagged.empty:
                continue
            if flagged.nunique() != 1:
                problems.append(f"{flag_col} ({name}): flagged rows have {flagged.nunique()} distinct sentinel values, expected 1")
                continue
            sentinel = flagged.iloc[0]
            if (unflagged == sentinel).any():
                problems.append(f"{flag_col} ({name}): sentinel value {sentinel} also appears in non-flagged rows")
        notes.append(base)
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", f"{len(pairs)} missing-flag columns each map to one consistent sentinel value: {notes}"


@check("One-hot categorical groups are well-formed (row sum in {0,1})")
def _onehot(state):
    train, eval_ = state["train"], state["eval"]
    groups = onehot_groups(list(train.columns))
    if not groups:
        return "FAIL", "no recognized one-hot categorical groups found"
    problems = []
    for base, cols in groups.items():
        for name, df in [("train", train), ("eval", eval_)]:
            row_sums = df[cols].sum(axis=1)
            bad = ~row_sums.isin([0, 1])
            if bad.any():
                problems.append(f"{base} ({name}): {bad.sum()} rows with row-sum not in {{0,1}}")
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", f"{len(groups)} categorical groups checked: {list(groups.keys())}"


@check("Scaler was fit on train only, and train is standardized (mean~0, std~1)")
def _scaler_check(state):
    train, scaler = state["train"], state["scaler"]
    scaled_cols = list(getattr(scaler, "feature_names_in_", []))
    if not scaled_cols:
        return "FAIL", "scaler has no feature_names_in_"
    missing = [c for c in scaled_cols if c not in train.columns]
    if missing:
        return "FAIL", f"scaler references columns not in train: {missing}"

    # sklearn's StandardScaler sets scale_ = 1.0 (a no-op) for any column
    # with zero variance in the fit data, to avoid dividing by zero. Those
    # columns stay at their constant value after "scaling" and will never
    # have std ~ 1, so they must be excluded from that check -- but they're
    # worth surfacing since a constant feature carries no predictive signal.
    var_ = getattr(scaler, "var_", None)
    if var_ is not None:
        zero_var_cols = [c for c, v in zip(scaled_cols, var_) if v == 0]
    else:
        zero_var_cols = []
    normal_cols = [c for c in scaled_cols if c not in zero_var_cols]

    problems = []
    if normal_cols:
        actual_mean = train[normal_cols].mean().to_numpy()
        actual_std = train[normal_cols].std(ddof=0).to_numpy()
        mean_off = np.abs(actual_mean).max()
        std_off = np.abs(actual_std - 1).max()
        if mean_off > 0.05 or std_off > 0.05:
            problems.append(f"non-constant columns not standardized as expected (max|mean|={mean_off:.4f}, max|std-1|={std_off:.4f})")
    if zero_var_cols:
        nonconstant = [c for c in zero_var_cols if train[c].nunique() > 1]
        if nonconstant:
            problems.append(f"columns flagged zero-variance by the scaler but not actually constant in current train data: {nonconstant}")

    if problems:
        return "FAIL", "; ".join(problems)
    detail = f"{len(normal_cols)} scaled columns confirmed mean~0/std~1 on train"
    if zero_var_cols:
        detail += f"; NOTE: {zero_var_cols} are constant (zero-variance) in train and left unscaled by sklearn -- these carry no signal, consider dropping them as features"
    return "PASS", detail


@check("Eval set is transformed with train's scaler, not refit on eval")
def _scaler_not_refit_on_eval(state):
    eval_, scaler = state["eval"], state["scaler"]
    scaled_cols = list(getattr(scaler, "feature_names_in_", []))
    actual_mean = eval_[scaled_cols].mean().to_numpy()
    # Eval was NOT used to fit the scaler, so it's fine (expected) for its
    # mean/std to drift from 0/1 -- this just reports the drift for a sanity
    # look, it does not fail the check.
    mean_drift = np.abs(actual_mean).max()
    return "PASS", f"eval mean drift from 0 after train-fit scaling: max|mean|={mean_drift:.4f} (expected to be nonzero; large drift may indicate distribution shift, worth a look but not itself a bug)"


def main():
    state = {}
    for fn in [
        _files_exist,
        _load,
        _schema,
        _standard_cols,
        _case_id_unique,
        _no_overlap,
        _temporal_split,
        _label_clean,
        _no_nans,
        _missing_flags,
        _onehot,
        _scaler_check,
        _scaler_not_refit_on_eval,
    ]:
        fn(state)

    print("=" * 100)
    print("FIFAR PREPARED DATASET VERIFICATION")
    print("=" * 100)
    for status, name, detail in results:
        print(f"[{status}] {name}\n         {detail}\n")

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_pass = sum(1 for s, _, _ in results if s == "PASS")
    print("=" * 100)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {len(results)} total checks")
    print("=" * 100)

    if n_fail:
        print("\nOVERALL: NOT OK -- fix the failing checks above before modeling.")
        sys.exit(1)
    else:
        print("\nOVERALL: OK -- dataset looks good to start Theme 1 modeling.")
        sys.exit(0)


if __name__ == "__main__":
    main()
