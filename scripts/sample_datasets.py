"""Sample large sharded molpile_atoms_{n} SMILES datasets down to a fixed size.

Two sampling methods are supported:

* ``random``  - a plain uniform random subset of the dataset.
* ``diverse`` - maximum diversity picking: pick maximally diverse cluster
  centers with RDKit's ``MaxMinPicker``, assign the remaining molecules to
  their nearest center and take an even slice from every cluster. Missing
  molecules (small clusters) are topped up randomly.

Each input dataset ``datasets/smiles/molpile_atoms_{n}`` (a directory of
parquet shards with columns ``id`` / ``smiles``) produces an output dataset
``datasets/smiles/molpile_atoms_{n}_samples_{method}`` written as parquet
shards with the same schema.

Datasets are sampled in parallel (one worker process per dataset) so the whole
range finishes faster.

Example:
    uv run scripts/sample_datasets.py \
        --min-atoms 10 --max-atoms 25 \
        --subset-size 100000 --method both --workers 8
"""

import glob
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import MaxMinPicker
from tqdm import tqdm

_SCHEMA = pa.schema([("id", pa.int64()), ("smiles", pa.string())])
SEED = 0


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def read_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Read every parquet shard of a dataset into a single DataFrame."""
    shards = sorted(glob.glob(str(dataset_dir / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {dataset_dir}")
    table = pq.read_table(shards, columns=["id", "smiles"])
    return table.to_pandas()


def write_dataset(df: pd.DataFrame, out_dir: Path, shard_size: int) -> None:
    """Write a DataFrame as parquet shards (same layout as the inputs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # start from a clean directory so re-runs don't leave stale shards behind
    for stale in out_dir.glob("*.parquet"):
        stale.unlink()

    df = df.reset_index(drop=True)
    for shard_idx, start in enumerate(range(0, len(df), shard_size)):
        chunk = df.iloc[start : start + shard_size]
        table = pa.Table.from_pandas(chunk, schema=_SCHEMA, preserve_index=False)
        pq.write_table(table, out_dir / f"s{shard_idx:04d}.parquet")


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #
def compute_fingerprints(
    smiles: list[str], desc: str = "fingerprints"
) -> tuple[list, list[int]]:
    """Compute Morgan fingerprints, returning (fps, valid_positions).

    ``valid_positions[i]`` is the index into ``smiles`` that produced
    ``fps[i]``. Molecules that fail to parse are dropped.
    """
    gen = GetMorganGenerator()
    fps: list = []
    valid_positions: list[int] = []
    for pos in tqdm(range(len(smiles)), desc=desc, unit="mol", leave=False):
        mol = Chem.MolFromSmiles(smiles[pos])
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        valid_positions.append(pos)
    return fps, valid_positions


# --------------------------------------------------------------------------- #
# Sampling methods
# --------------------------------------------------------------------------- #
def random_sample(df: pd.DataFrame, subset_size: int) -> pd.DataFrame:
    """Uniform random subset (without replacement)."""
    n = min(subset_size, len(df))
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idxs)].reset_index(drop=True)


def diverse_sample(
    df: pd.DataFrame, subset_size: int, threshold: float = 0.9
) -> pd.DataFrame:
    """Maximum diversity subset via MaxMinPicker centers + cluster slicing.

    All work happens in "valid space" (positions into the list of molecules
    that parse successfully); results are mapped back to ``df`` rows at the end.
    """
    if len(df) <= subset_size:
        return df.reset_index(drop=True)

    smiles = df["smiles"].tolist()
    fps, valid_positions = compute_fingerprints(smiles, desc="fingerprints")
    n_valid = len(fps)

    if n_valid <= subset_size:
        # not enough valid molecules to be picky; keep them all
        return df.iloc[valid_positions].reset_index(drop=True)

    # pick maximally diverse centers (indices are in valid space, 0..n_valid-1)
    picker = MaxMinPicker()
    centers, _ = picker.LazyBitVectorPickWithThreshold(
        fps, poolSize=n_valid, pickSize=n_valid, threshold=threshold, seed=SEED
    )
    centers = list(centers)
    n_centers = len(centers)

    # if we already have at least as many centers as we want, they are the most
    # diverse molecules available -> just take the first subset_size of them
    if n_centers >= subset_size:
        selected_valid = centers[:subset_size]
        df_positions = [valid_positions[i] for i in selected_valid]
        return df.iloc[df_positions].reset_index(drop=True)

    # otherwise assign every non-center molecule to its nearest center and take
    # an even slice from each cluster
    clusters: dict[int, list[int]] = {c: [c] for c in centers}
    center_set = set(centers)
    pool = [i for i in range(n_valid) if i not in center_set]

    rng = random.Random(SEED)
    rng.shuffle(pool)

    n_mols_per_center = max(0, (subset_size // n_centers) - 1)
    center_fps = [fps[c] for c in centers]

    if n_mols_per_center > 0:
        for i in tqdm(pool, desc="cluster assign", unit="mol", leave=False):
            sims = BulkTanimotoSimilarity(fps[i], center_fps)
            best = centers[int(np.argmax(sims))]
            if len(clusters[best]) <= n_mols_per_center:  # +1 for the center
                clusters[best].append(i)

    selected_valid = [i for members in clusters.values() for i in members]
    df_positions = [valid_positions[i] for i in selected_valid]
    subset_df = df.iloc[df_positions]

    # top up randomly from unused valid molecules if clusters were too small
    if len(subset_df) < subset_size:
        selected_set = set(selected_valid)
        remaining = [
            valid_positions[i] for i in range(n_valid) if i not in selected_set
        ]
        needed = min(subset_size - len(subset_df), len(remaining))
        fill_rng = np.random.default_rng(SEED)
        fill = fill_rng.choice(remaining, size=needed, replace=False)
        subset_df = pd.concat([subset_df, df.iloc[fill]])

    return subset_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-dataset worker
# --------------------------------------------------------------------------- #
def sample_one(
    atoms: int,
    method: str,
    input_dir: str,
    output_dir: str,
    subset_size: int,
    threshold: float,
    shard_size: int,
) -> tuple[int, str, int]:
    RDLogger.DisableLog("rdApp.*")

    dataset_dir = Path(input_dir) / f"molpile_atoms_{atoms}"
    df = read_dataset(dataset_dir)

    if method == "random":
        subset = random_sample(df, subset_size)
    elif method == "diverse":
        subset = diverse_sample(df, subset_size, threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    out_dir = Path(output_dir) / f"molpile_atoms_{atoms}_samples_{method}"
    write_dataset(subset, out_dir, shard_size)
    return atoms, method, len(subset)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--input-dir", default="datasets/smiles", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--output-dir", default="datasets/smiles", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--min-atoms", default=10, show_default=True, type=int)
@click.option("--max-atoms", default=25, show_default=True, type=int)
@click.option("--subset-size", default=100_000, show_default=True, type=int,
              help="Target number of molecules per output dataset.")
@click.option("--method", type=click.Choice(["random", "diverse", "both"]),
              default="both", show_default=True)
@click.option("--threshold", default=0.9, show_default=True, type=float,
              help="Tanimoto distance threshold for MaxMinPicker (diverse only).")
@click.option("--workers", default=4, show_default=True, type=int,
              help="Number of datasets sampled in parallel.")
@click.option("--shard-size", default=5_000, show_default=True, type=int)
def main(input_dir, output_dir, min_atoms, max_atoms, subset_size, method,
         threshold, workers, shard_size) -> None:
    methods = ["random", "diverse"] if method == "both" else [method]

    jobs = []
    for atoms in range(min_atoms, max_atoms + 1):
        dataset_dir = Path(input_dir) / f"molpile_atoms_{atoms}"
        if not dataset_dir.is_dir():
            click.echo(f"SKIP: {dataset_dir} does not exist")
            continue
        for m in methods:
            jobs.append((atoms, m))

    if not jobs:
        click.echo("No datasets to sample.")
        return

    click.echo(
        f"Sampling {len(jobs)} job(s) "
        f"(atoms {min_atoms}-{max_atoms}, methods {methods}, "
        f"size {subset_size}) with {workers} worker(s)."
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                sample_one, atoms, m, input_dir, output_dir,
                subset_size, threshold, shard_size,
            ): (atoms, m)
            for atoms, m in jobs
        }
        for future in as_completed(futures):
            atoms, m = futures[future]
            try:
                a, mm, n = future.result()
                click.echo(f"DONE: molpile_atoms_{a}_samples_{mm} ({n} molecules)")
            except Exception as exc:  # noqa: BLE001
                click.echo(f"FAILED: molpile_atoms_{atoms}_samples_{m} -> {exc}")

    click.echo("All sampling jobs finished.")


if __name__ == "__main__":
    main()
