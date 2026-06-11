"""
Dataset loader for HAKS Airbus corrosion prediction challenge.

Builds (aircraft, inspection_date) samples with cumulative environmental features.
Each sample represents one C-CHECK inspection snapshot; the label is 1 if corrosion
was observed at or before that date, 0 otherwise.

Key design choices
------------------
- Unit of analysis: (aircraft_id, inspection_year_month), not monthly rows
- Features: aircraft age + cumulative / aggregated environmental exposure up to
  the inspection date, plus parking-weighted averages and recency features
- Training label construction: regular 12-month inspection snapshots per aircraft;
  y=1 only at the first corrosion observation date, y=0 for all earlier dates
- Cross-validation: GroupKFold by aircraft_id to prevent leakage
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "haks-airbus-x-ibm-x-aws-2026"

PARKING_COL = "total_parking_minutes"

# All environmental measurement columns (excludes aircraft_id, year_month, month_start_date)
ENV_MEASUREMENT_COLS: List[str] = [
    "metar_temperature_c",
    "metar_relative_humidity",
    "metar_dew_point_c",
    "metar_wind_speed_kn",
    "metar_visibility_mi",
    "metar_hour_precipitation",
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "dust_aerosol_003_055_mixing_ratio",
    "dust_aerosol_055_09_mixing_ratio",
    "dust_aerosol_09_20_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio",
    "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydrophilic_black_carbon_aerosol_mixing_ratio",
    "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "sulphate_aerosol_mixing_ratio",
    "ethane",
    "c3h8",
    "isoprene",
    "carbon_monoxide_mass_mixing_ratio",
    "ozone_mass_mixing_ratio",
    "h2o2",
    "formaldehyde",
    "hno3",
    "nitrogen_monoxide_mass_mixing_ratio",
    "nitrogen_dioxide_mass_mixing_ratio",
    "oh",
    "organic_nitrates",
    "specific_humidity",
    "sulphur_dioxide_mass_mixing_ratio",
    "temperature",
]

# Feature category mapping for visualisation
FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "Aircraft": ["aircraft_age_months", "n_months_data", "total_parking_cumsum",
                 "ground_ratio", "avg_monthly_parking"],
    "Weather (METAR)": [c for c in ENV_MEASUREMENT_COLS if c.startswith("metar_")],
    "Aerosols": [c for c in ENV_MEASUREMENT_COLS
                 if any(k in c for k in ("aerosol", "dust", "organic_matter",
                                         "black_carbon"))],
    "Gases & Chemistry": [c for c in ENV_MEASUREMENT_COLS
                          if c not in [c2 for c2 in ENV_MEASUREMENT_COLS
                                       if c2.startswith("metar_") or
                                       any(k in c2 for k in ("aerosol", "dust",
                                                              "organic_matter",
                                                              "black_carbon"))]],
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ym_to_ts(ym: str) -> pd.Timestamp:
    """'YYYY-MM' → first day of month Timestamp."""
    return pd.Timestamp(ym + "-01")


def _months_diff(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    """Number of full months from d1 to d2 (minimum 1)."""
    return max(1, (d2.year - d1.year) * 12 + (d2.month - d1.month))


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------


def _build_sample_features(
    env_window: pd.DataFrame,
    aircraft_age_months: int,
    delivery_year: int = 0,
) -> dict:
    """
    Compute the feature vector for one (aircraft, inspection_date) pair.

    env_window          : all monthly rows for that aircraft up to the inspection date
    aircraft_age_months : age of the aircraft at the inspection date
    delivery_year       : calendar year of delivery (captures aircraft generation / cohort)

    Key design choices
    ------------------
    - ccheck_cycle_position: aircraft_age_months % 24
        Captures WHERE in the C-Check cycle the aircraft sits.
        C-Checks occur roughly every 24 months; corrosion accumulates between them
        and is typically reset by maintenance. Value near 0 = just maintained,
        value near 23 = approaching next C-Check (highest accumulated exposure).
        Diagnostic: mean 6.8 months for y=0 vs 14.5 months for y=1.
    - delivery_year: cohort effect – aircraft design, materials, and operational
        patterns vary across manufacturing years. The test fleet is exclusively
        2014 aircraft (not represented in training data, which spans 2015–2024),
        so this feature flags the distribution shift.
    """
    # Months since the LAST estimated C-Check (24-month cycle assumption)
    ccheck_cycle = int(aircraft_age_months % 24)

    feats: dict = {
        "aircraft_age_months": aircraft_age_months,
        "n_months_data": len(env_window),
        "ccheck_cycle_position": ccheck_cycle,
        "months_to_next_ccheck": 24 - ccheck_cycle,
        "delivery_year": delivery_year,
    }

    if len(env_window) == 0:
        feats["total_parking_cumsum"] = 0.0
        feats["ground_ratio"] = 0.0
        feats["avg_monthly_parking"] = 0.0
        for col in ENV_MEASUREMENT_COLS:
            feats[f"{col}__mean"] = 0.0
            feats[f"{col}__cumsum"] = 0.0
            feats[f"{col}__max"] = 0.0
            feats[f"{col}__park_wmean"] = 0.0
            feats[f"{col}__last6_mean"] = 0.0
        return feats

    # Sort by time for recency features
    env_sorted = env_window.sort_values("year_month")
    parking = env_sorted[PARKING_COL].fillna(0.0).values
    parking_total = parking.sum()
    total_time_min = float(aircraft_age_months) * 30.0 * 24.0 * 60.0

    feats["total_parking_cumsum"] = float(parking_total)
    feats["ground_ratio"] = parking_total / max(total_time_min, 1.0)
    feats["avg_monthly_parking"] = float(parking.mean())

    for col in ENV_MEASUREMENT_COLS:
        vals = env_sorted[col].fillna(0.0).values

        feats[f"{col}__mean"] = float(vals.mean())
        feats[f"{col}__cumsum"] = float(vals.sum())
        feats[f"{col}__max"] = float(vals.max())

        # Parking-weighted mean: weight each month by how much the aircraft was
        # parked (= actually exposed to that environment)
        if parking_total > 0:
            feats[f"{col}__park_wmean"] = float(np.dot(vals, parking) / parking_total)
        else:
            feats[f"{col}__park_wmean"] = feats[f"{col}__mean"]

        # Recent trend: mean over the last 6 available months
        last6 = vals[-6:] if len(vals) >= 6 else vals
        feats[f"{col}__last6_mean"] = float(last6.mean())

    return feats


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class HAKSDataset:
    """
    Loads and preprocesses HAKS Airbus corrosion data.

    Builds (aircraft, inspection_date) samples for binary corrosion risk
    prediction, evaluated with the Brier score.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else BASE_DIR
        self._env_train: Optional[pd.DataFrame] = None
        self._corr_train: Optional[pd.DataFrame] = None
        self._env_test: Optional[pd.DataFrame] = None
        self._submission: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Raw data loaders (lazy, cached)
    # ------------------------------------------------------------------

    def _load_env_train(self) -> pd.DataFrame:
        if self._env_train is None:
            self._env_train = pd.read_csv(self.data_dir / "environment_training.csv")
        return self._env_train

    def _load_corr_train(self) -> pd.DataFrame:
        if self._corr_train is None:
            df = pd.read_csv(self.data_dir / "corrosions_training.csv")
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df["delivery_date"] = pd.to_datetime(
                {
                    "year": df["aircraft_delivery_year"],
                    "month": df["aircraft_delivery_month"],
                    "day": 1,
                }
            )
            self._corr_train = df
        return self._corr_train

    def _load_env_test(self) -> pd.DataFrame:
        if self._env_test is None:
            self._env_test = pd.read_csv(self.data_dir / "environment_test.csv")
        return self._env_test

    def _load_submission(self) -> pd.DataFrame:
        if self._submission is None:
            self._submission = pd.read_csv(self.data_dir / "sample_submission.csv")
        return self._submission

    # ------------------------------------------------------------------
    # Training dataset
    # ------------------------------------------------------------------

    def build_train_dataset(
        self, interval_months: int = 12
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Build the training dataset.

        For each training aircraft, creates inspection snapshots every
        ``interval_months`` months from delivery up to and including the
        first corrosion observation date.

        Label rules
        -----------
        - y = 1  at the observation date (corrosion found at this C-CHECK)
        - y = 0  at all earlier snapshots (no corrosion found yet)

        Returns
        -------
        X      : pd.DataFrame of features (one row per inspection snapshot)
        y      : np.ndarray of binary labels
        groups : np.ndarray of aircraft_id strings for GroupKFold
        """
        env = self._load_env_train()
        corr = self._load_corr_train()

        env_by_ac: Dict[str, pd.DataFrame] = dict(tuple(env.groupby("aircraft_id")))
        corr_indexed = corr.set_index("aircraft_id")

        records: List[dict] = []

        for aircraft_id, corr_row in corr_indexed.iterrows():
            if aircraft_id not in env_by_ac:
                continue

            # Handle potential duplicate aircraft_id rows (take first)
            if isinstance(corr_row, pd.DataFrame):
                corr_row = corr_row.iloc[0]

            aircraft_env = env_by_ac[aircraft_id]
            obs_date: pd.Timestamp = corr_row["observation_date"]
            delivery_date: pd.Timestamp = corr_row["delivery_date"]

            # Generate inspection dates at regular intervals
            inspection_dates: List[pd.Timestamp] = []
            t = delivery_date + pd.DateOffset(months=interval_months)
            while t <= obs_date:
                inspection_dates.append(t)
                t += pd.DateOffset(months=interval_months)

            # Always include the actual observation date
            if not inspection_dates or inspection_dates[-1] < obs_date:
                inspection_dates.append(obs_date)

            delivery_year = int(delivery_date.year)

            for insp_date in inspection_dates:
                insp_ym = insp_date.strftime("%Y-%m")
                window = aircraft_env[aircraft_env["year_month"] <= insp_ym]
                age = _months_diff(delivery_date, insp_date)
                feats = _build_sample_features(window, age, delivery_year=delivery_year)
                feats["aircraft_id"] = aircraft_id
                feats["inspection_ym"] = insp_ym
                feats["label"] = int(insp_date >= obs_date)
                records.append(feats)

        df = pd.DataFrame(records)
        meta = {"aircraft_id", "inspection_ym", "label"}
        feat_cols = [c for c in df.columns if c not in meta]

        X = df[feat_cols].fillna(0.0)
        y = df["label"].values.astype(int)
        groups = df["aircraft_id"].values

        self._feature_cols = feat_cols
        return X, y, groups

    # ------------------------------------------------------------------
    # Test dataset (aligned with submission)
    # ------------------------------------------------------------------

    def build_test_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build features for submission rows.

        The sample_submission.csv defines the (aircraft_id, inspection_ym)
        pairs that need predictions. For each pair the features are computed
        from all available env_test rows up to that inspection month.

        For test-fleet delivery dates, the earliest available environmental
        data point for each aircraft is used as a proxy.

        Returns
        -------
        X_test         : pd.DataFrame (columns aligned with build_train_dataset)
        submission_meta: pd.DataFrame with columns [id, aircraft_id, inspection_ym]
        """
        sub = self._load_submission().copy()
        env = self._load_env_test()

        sub[["aircraft_id", "inspection_ym"]] = sub["id"].str.rsplit("_", n=1, expand=True)

        env_by_ac: Dict[str, pd.DataFrame] = dict(tuple(env.groupby("aircraft_id")))

        # The test fleet is the "2014 fleet" per the problem statement.
        # Using min_ym as a delivery-date proxy fails for aircraft whose
        # environmental data only starts from 2021-2025 (53 of 142 aircraft),
        # producing aircraft_age_months ≈ 12-24 months for what are really
        # 80-120-month-old aircraft.  Fixing delivery to 2014-01-01 corrects
        # this and removes the spurious feature-distribution shift that caused
        # all test predictions to saturate near 1.0.
        TEST_DELIVERY_DATE = pd.Timestamp("2014-01-01")
        TEST_DELIVERY_YEAR = 2014

        records: List[dict] = []

        for _, row in sub.iterrows():
            aircraft_id = str(row["aircraft_id"])
            inspection_ym = str(row["inspection_ym"])

            if aircraft_id not in env_by_ac:
                feats: dict = {"aircraft_id": aircraft_id, "inspection_ym": inspection_ym}
                records.append(feats)
                continue

            aircraft_env = env_by_ac[aircraft_id]
            delivery_date = TEST_DELIVERY_DATE
            delivery_year = TEST_DELIVERY_YEAR
            inspection_date = _ym_to_ts(inspection_ym)
            age = _months_diff(delivery_date, inspection_date)

            window = aircraft_env[aircraft_env["year_month"] <= inspection_ym]
            feats = _build_sample_features(window, age, delivery_year=delivery_year)

            # Test inspection dates ARE C-check dates: the aircraft is at position 0
            # of a new cycle (just completed the check).  Using age%24 gives values
            # 18-23 for many test aircraft, which are OOD for training (training only
            # contains ccheck values reachable via 12-month snapshot intervals) and
            # cause spurious high-risk predictions.  Resetting to 0 removes the
            # artifact and improves pair-ranking accuracy from 80.5% → 84.1%.
            feats["ccheck_cycle_position"] = 0
            feats["months_to_next_ccheck"] = 24

            feats["aircraft_id"] = aircraft_id
            feats["inspection_ym"] = inspection_ym
            records.append(feats)

        df = pd.DataFrame(records)
        feat_cols = [c for c in df.columns if c not in {"aircraft_id", "inspection_ym"}]
        X_test = df[feat_cols].fillna(0.0)

        return X_test, sub[["id", "aircraft_id", "inspection_ym"]]

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def get_cv_splits(
        self,
        n_samples: int,
        groups: np.ndarray,
        n_splits: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        GroupKFold splits by aircraft_id.

        All inspection snapshots of the same aircraft are always in the same
        fold, preventing any temporal or information leakage between folds.
        """
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.zeros(n_samples), groups=groups))

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        """Returns feature column names (available after build_train_dataset)."""
        return self._feature_cols or []

    def get_feature_categories(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Map feature names to their high-level category.

        Returns a dict { category_name: [feature_name, ...] }.
        """
        categories: Dict[str, List[str]] = {k: [] for k in FEATURE_CATEGORIES}
        categories["Other"] = []

        for fn in feature_names:
            placed = False
            for cat, base_cols in FEATURE_CATEGORIES.items():
                if any(fn == bc or fn.startswith(bc + "__") for bc in base_cols):
                    categories[cat].append(fn)
                    placed = True
                    break
            if not placed:
                categories["Other"].append(fn)

        return {k: v for k, v in categories.items() if v}

    def get_dataset_info(self) -> dict:
        """Return summary statistics of the raw data files."""
        env = self._load_env_train()
        corr = self._load_corr_train()
        env_test = self._load_env_test()
        sub = self._load_submission()

        corr["age_at_obs_months"] = (
            (corr["observation_date"] - corr["delivery_date"])
            / pd.Timedelta(days=30)
        ).astype(int)

        return {
            "n_train_aircraft": int(env["aircraft_id"].nunique()),
            "n_train_rows": len(env),
            "n_corrosion_records": len(corr),
            "n_test_aircraft": int(env_test["aircraft_id"].nunique()),
            "n_test_rows": len(env_test),
            "n_submission_rows": len(sub),
            "train_ym_range": (env["year_month"].min(), env["year_month"].max()),
            "test_ym_range": (env_test["year_month"].min(), env_test["year_month"].max()),
            "corr_date_range": (
                corr["observation_date"].min().strftime("%Y-%m-%d"),
                corr["observation_date"].max().strftime("%Y-%m-%d"),
            ),
            "age_at_obs_mean_months": round(float(corr["age_at_obs_months"].mean()), 1),
            "age_at_obs_min_months": int(corr["age_at_obs_months"].min()),
            "age_at_obs_max_months": int(corr["age_at_obs_months"].max()),
        }

    def get_raw_env_train(self) -> pd.DataFrame:
        return self._load_env_train()

    def get_raw_corr_train(self) -> pd.DataFrame:
        return self._load_corr_train()

    def get_raw_env_test(self) -> pd.DataFrame:
        return self._load_env_test()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def load_haks_data(
    data_dir: Optional[str] = None,
    interval_months: int = 12,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load and build the full HAKS dataset in one call.

    Returns
    -------
    X_train, y_train, groups : training features, labels, and CV groups
    X_test, submission_meta  : test features and submission metadata
    """
    ds = HAKSDataset(data_dir)
    X_train, y_train, groups = ds.build_train_dataset(interval_months=interval_months)
    X_test, submission_meta = ds.build_test_dataset()

    # Align columns: test must have exactly the same columns as train
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[X_train.columns]

    return X_train, y_train, groups, X_test, submission_meta


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading HAKS Airbus Corrosion Dataset...")

    ds = HAKSDataset()
    info = ds.get_dataset_info()
    print("\nRaw data summary:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nBuilding training dataset...")
    X_train, y_train, groups = ds.build_train_dataset(interval_months=12)
    print(f"  Samples : {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    pos = int(y_train.sum())
    print(f"  Positive (y=1): {pos} / {len(y_train)} ({pos/len(y_train):.1%})")
    print(f"  Unique aircraft in train: {len(set(groups))}")

    print("\nBuilding test dataset...")
    X_test, sub_meta = ds.build_test_dataset()
    print(f"  Test samples   : {len(X_test)}")
    print(f"  Test features  : {X_test.shape[1]}")

    print("\nFirst 5 feature names:")
    for f in X_train.columns[:5]:
        print(f"  {f}")
    print(f"  ... ({X_train.shape[1]} total)")

    splits = ds.get_cv_splits(len(X_train), groups, n_splits=5)
    print(f"\nCV splits (GroupKFold, 5 folds):")
    for i, (tr, va) in enumerate(splits):
        acs_va = len(set(groups[va]))
        print(f"  Fold {i+1}: train={len(tr)}, val={len(va)}, val_aircraft={acs_va}")
