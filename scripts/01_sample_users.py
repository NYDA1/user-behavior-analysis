"""01 · Sample complete users from the raw dataset.

Streams the raw CSV twice: first collects every user_id, then extracts all
rows belonging to a random sample of ``SAMPLE_SIZE`` users. Because the raw
file is sorted by user_id, extracting a user's rows is exact — the sample
holds the complete behavior history of each selected user, nothing else.

Peak memory stays well under a few hundred MB regardless of dataset size.

Outputs:
    data/processed/sampled_user_ids.csv   the selected user ids (reproducible)
    data/processed/user_sample.parquet    all events of the sampled users
"""

import random

import pandas as pd

from user_behavior_analysis.config import (
    COLUMNS,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    READ_DTYPES,
    SAMPLE_SIZE,
    SEED,
)
from user_behavior_analysis.io import log_step, timed

CHUNK_SIZE = 1_000_000


@timed("scan raw data for all user ids")
def collect_user_ids() -> set:
    user_ids = set()
    for i, chunk in enumerate(
        pd.read_csv(
            RAW_DATA_PATH,
            chunksize=CHUNK_SIZE,
            names=COLUMNS,
            header=None,
            usecols=["user_id"],
        )
    ):
        user_ids.update(chunk["user_id"].unique().tolist())
        if (i + 1) % 10 == 0:
            log_step(f"  pass 1: scanned {CHUNK_SIZE * (i + 1):,} rows, {len(user_ids):,} users")
    return user_ids


@timed("sample users")
def select_users(user_ids: set) -> set:
    # sorted() keeps the selection reproducible across Python versions/hashes.
    rng = random.Random(SEED)
    selected = set(rng.sample(sorted(user_ids), SAMPLE_SIZE))
    out = pd.DataFrame({"user_id": sorted(selected)})
    out.to_csv(PROCESSED_DIR / "sampled_user_ids.csv", index=False)
    return selected


@timed("extract sampled users' events")
def extract_events(selected: set) -> pd.DataFrame:
    frames = []
    collected = 0
    for i, chunk in enumerate(
        pd.read_csv(
            RAW_DATA_PATH,
            chunksize=CHUNK_SIZE,
            names=COLUMNS,
            header=None,
            dtype=READ_DTYPES,
        )
    ):
        mask = chunk["user_id"].isin(selected)
        if mask.any():
            frames.append(chunk.loc[mask])
            collected += int(mask.sum())
        if (i + 1) % 10 == 0:
            log_step(f"  pass 2: scanned {CHUNK_SIZE * (i + 1):,} rows, collected {collected:,} events")
    return pd.concat(frames, ignore_index=True)


def validate(df: pd.DataFrame, selected: set) -> None:
    n_users = df["user_id"].nunique()
    assert n_users == len(selected), f"expected {len(selected)} users, got {n_users}"
    assert set(df["user_id"].unique().tolist()) == selected, "user ids differ from selection"

    dist = df["behavior_type"].value_counts(normalize=True) * 100
    log_step(
        f"sample: {len(df):,} events, {n_users:,} users, "
        f"{len(df) / n_users:.0f} events/user"
    )
    for b in ["pv", "cart", "fav", "buy"]:
        pct = dist.get(b, 0.0)
        marker = ""
        # Reference distribution of the public dataset (approx): pv ~89.6%,
        # cart ~5.5%, fav ~2.5%, buy ~2.4%. A good sample is within ~1.5 pp.
        ref = {"pv": 89.6, "cart": 5.5, "fav": 2.5, "buy": 2.4}[b]
        if abs(pct - ref) > 1.5:
            marker = "  <-- deviates from public dataset by >1.5pp"
        log_step(f"  {b}: {pct:.2f}%{marker}")


def main() -> None:
    log_step("01 · sampling complete users")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log_step(f"raw data: {RAW_DATA_PATH} ({RAW_DATA_PATH.stat().st_size / 1e9:.2f} GB)")

    user_ids = collect_user_ids()
    log_step(f"full dataset has {len(user_ids):,} users")
    if len(user_ids) < SAMPLE_SIZE:
        raise SystemExit(f"dataset has fewer than {SAMPLE_SIZE} users; aborting")

    selected = select_users(user_ids)
    log_step(f"selected {len(selected)} users (seed={SEED})")

    df = extract_events(selected)
    validate(df, selected)

    path = PROCESSED_DIR / "user_sample.parquet"
    df.to_parquet(path, index=False)
    log_step(f"sample written -> {path}")


if __name__ == "__main__":
    main()
