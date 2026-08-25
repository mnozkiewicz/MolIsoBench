"""Count embedding collisions for one model across all its datasets.

For the given model (a subdirectory of the embeddings dir, i.e. a --run-name from
the embedding step), this iterates every dataset, counts collisions at each radius,
and writes a tidy per-model Parquet file:

    columns: model, dataset, radius, collisions, pairs, rate   (one row per dataset x radius)

The result is written after every dataset (crash-safe) and the run is resumable:
datasets already present in the output file are skipped by default.

For the whole-benchmark analysis, concatenate the per-model files:

    import glob, pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in glob.glob("datasets/collisions/*.parquet"))

Example:
    uv run scripts/count_collisions.py chemformer
    uv run scripts/count_collisions.py mol-r-gcn --backend torch --radii 1e-4,1e-3,1e-2
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
import pandas as pd
from natsort import natsorted

from src.collisions import BACKENDS, DEFAULT_RADII, count_dataset_collisions


@click.command()
@click.argument("model_name")
@click.option("--embeddings-dir", default="datasets/embeddings", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--output-dir", default="datasets/collisions", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--backend", type=click.Choice(list(BACKENDS)), default="faiss", show_default=True,
              help="Collision counter. faiss is exact + memory-light (good default); "
                   "torch is fastest on GPU.")
@click.option("--radii", default=",".join(f"{r:g}" for r in DEFAULT_RADII), show_default=True,
              help="Comma-separated cosine-distance thresholds, evaluated in one pass.")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip datasets already present in the output file.")
def main(model_name: str, embeddings_dir: str, output_dir: str, backend: str, radii: str, resume: bool) -> None:
    radii_list = [float(x) for x in radii.split(",") if x.strip()]

    model_dir = Path(embeddings_dir) / model_name
    if not model_dir.is_dir():
        raise click.ClickException(f"No embeddings for model '{model_name}' at {model_dir}")

    datasets = [d for d in natsorted(model_dir.iterdir(), key=lambda p: p.name) if d.is_dir()]
    if not datasets:
        raise click.ClickException(f"No dataset subdirectories under {model_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.parquet"

    rows: list[dict] = []
    done: set[str] = set()
    if resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        rows = prev.to_dict("records")
        done = set(prev["dataset"].unique())
        click.echo(f"Resuming from {out_path}: {len(done)} dataset(s) already done.")

    click.echo(f"[{model_name}] {len(datasets)} dataset(s), backend={backend}, radii={radii_list}")

    for d in datasets:
        if d.name in done:
            click.echo(f"  skip {d.name} (already done)")
            continue

        click.echo(f"  {d.name} ...", nl=False)
        totals = count_dataset_collisions(d, radii=radii_list, backend=backend)
        for r in radii_list:
            collisions, pairs = totals[r]
            rows.append({
                "model": model_name,
                "dataset": d.name,
                "radius": r,
                "collisions": collisions,
                "pairs": pairs,
                "rate": collisions / pairs if pairs > 0 else float("nan"),
            })
        # rewrite the (small) file after each dataset so a crash loses at most one
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        rate_1e3 = next((row["rate"] for row in rows[-len(radii_list):] if row["radius"] == radii_list[0]), float("nan"))
        click.echo(f" done (rate@{radii_list[0]:g}={rate_1e3:.2e})")

    click.echo(f"\nSaved {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
