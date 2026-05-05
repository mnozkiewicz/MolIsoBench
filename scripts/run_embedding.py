import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click


def collect_shards(input_dir: Path) -> list[Path]:
    shards = []
    for dataset_dir in sorted(input_dir.iterdir()):
        if dataset_dir.is_dir():
            shards.extend(sorted(dataset_dir.glob("shard_*.parquet")))
    return shards


def embed_one(model_dir: Path, shard_path: Path, extra_args: list[str]) -> tuple[str, bool]:
    result = subprocess.run(
        ["bash", "embed_model.sh", str(model_dir), str(shard_path), *extra_args],
        capture_output=False,
    )
    return str(shard_path), result.returncode == 0


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--input-dir", default="datasets/smiles", show_default=True, type=click.Path(file_okay=False))
@click.option("--workers", default=4, show_default=True, type=int)
@click.pass_context
def main(ctx, model_dir: str, input_dir: str, workers: int) -> None:
    model_dir_path = Path(model_dir)
    shards = collect_shards(Path(input_dir))

    if not shards:
        click.echo(f"No shards found in {input_dir}")
        return

    click.echo(f"Found {len(shards)} shard(s). Running with {workers} worker(s).")

    failed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(embed_one, model_dir_path, s, ctx.args): s
            for s in shards
        }
        for future in as_completed(futures):
            name, ok = future.result()
            if not ok:
                failed.append(name)
                click.echo(f"FAILED: {name}")

    if failed:
        click.echo(f"\n{len(failed)} shard(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        click.echo("\nAll shards embedded successfully.")


if __name__ == "__main__":
    main()
