import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click


def embed_one(model_dir: Path, input_path: Path, extra_args: list[str]) -> tuple[str, bool]:
    result = subprocess.run(
        ["bash", "embed_model.sh", str(model_dir), str(input_path), *extra_args],
        capture_output=False,
    )
    return input_path.name, result.returncode == 0


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--input-dir", default="datasets/smiles", show_default=True, type=click.Path(file_okay=False))
@click.option("--workers", default=4, show_default=True, type=int, help="Maximum number of concurrent processes")
@click.pass_context
def main(ctx, model_dir: str, input_dir: str, workers: int) -> None:
    model_dir_path = Path(model_dir)
    parquet_files = sorted(Path(input_dir).glob("*.parquet"))

    if not parquet_files:
        click.echo(f"No .parquet files found in {input_dir}")
        return

    click.echo(f"Found {len(parquet_files)} file(s). Running with {workers} worker(s).")

    failed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(embed_one, model_dir_path, f, ctx.args): f
            for f in parquet_files
        }
        for future in as_completed(futures):
            name, ok = future.result()
            if not ok:
                failed.append(name)
                click.echo(f"FAILED: {name}")

    if failed:
        click.echo(f"\n{len(failed)} file(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        click.echo("\nAll files embedded successfully.")


if __name__ == "__main__":
    main()
