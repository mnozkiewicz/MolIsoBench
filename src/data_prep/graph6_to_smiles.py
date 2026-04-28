from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

from src.providers.graph6 import Graph6Provider
from src.converters.graph_to_smiles import GraphToCarbonSMILESConverter


def _flush_shard(buffer: list, output_dir: Path, shard_idx: int, input_name: str) -> None:
    table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
    pq.write_table(table, output_dir / f"shard_{shard_idx:04d}.parquet")
    click.echo(f"[{input_name}] shard {shard_idx} ({len(buffer)} rows)")


def convert_graph6_to_smiles(
    input_path: str, output_dir: Path, shard_size: int = 5_000
) -> None:
    RDLogger.DisableLog("rdApp.*")

    output_dir.mkdir(parents=True, exist_ok=True)

    provider = Graph6Provider(input_path)
    converter = GraphToCarbonSMILESConverter()
    input_name = Path(input_path).name

    buffer = []
    shard_idx = 0

    for metadata, graph in provider:
        buffer.append({"id": metadata["id"], "smiles": converter.convert(graph)})

        if len(buffer) >= shard_size:
            _flush_shard(buffer, output_dir, shard_idx, input_name)
            buffer.clear()
            shard_idx += 1

    if buffer:
        _flush_shard(buffer, output_dir, shard_idx, input_name)

    click.echo(f"[{input_name}] Done -> {output_dir} ({shard_idx + 1} shard(s))")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="datasets/smiles", show_default=True)
@click.option("--shard-size", default=5_000, show_default=True, type=int)
def main(input: str, output_dir: str, shard_size: int) -> None:
    input_path = Path(input)
    convert_graph6_to_smiles(str(input_path), Path(output_dir) / input_path.stem, shard_size)


if __name__ == "__main__":
    main()
