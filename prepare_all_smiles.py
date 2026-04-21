import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import click


def convert_one(input_path: Path, output_dir: Path, batch_size: int) -> tuple[str, bool]:
    result = subprocess.run(
        [
            sys.executable, "-m", "src.data_prep.graph6_to_smiles",
            str(input_path),
            "--output-dir", str(output_dir),
            "--batch-size", str(batch_size),
        ],
        capture_output=False,
    )
    return input_path.name, result.returncode == 0


@click.command()
@click.option("--input-dir", default="datasets/g6", show_default=True, type=click.Path(file_okay=False))
@click.option("--output-dir", default="datasets/smiles", show_default=True, type=click.Path(file_okay=False))
@click.option("--workers", default=4, show_default=True, type=int, help="Maximum number of concurrent processes")
@click.option("--batch-size", default=10_000, show_default=True, type=int)
def main(input_dir: str, output_dir: str, workers: int, batch_size: int) -> None:
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    g6_files = sorted(input_dir_path.glob("*.g6"))
    if not g6_files:
        click.echo(f"No .g6 files found in {input_dir_path}")
        return

    click.echo(f"Found {len(g6_files)} file(s). Running with {workers} worker(s).")

    failed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_one, f, output_dir_path, batch_size): f
            for f in g6_files
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
        click.echo("\nAll files converted successfully.")


if __name__ == "__main__":
    main()
