from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

from src.providers.graph6 import Graph6Provider
from src.converters.graph_to_smiles import GraphToCarbonSMILESConverter


def convert_graph6_to_smiles(input_path: str, output_path: str, batch_size: int = 10_000) -> None:
    RDLogger.DisableLog("rdApp.*")

    provider = Graph6Provider(input_path)
    converter = GraphToCarbonSMILESConverter()

    writer = None
    buffer = []
    batch_counter = 0

    for metadata, graph in provider:
        smiles = converter.convert(graph)
        buffer.append({"id": metadata["id"], "smiles": smiles})

        if len(buffer) >= batch_size:
            table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            buffer.clear()
            batch_counter += 1
            click.echo(f"[{Path(input_path).name}] Processed {batch_counter} batches")

    if buffer:
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    click.echo(f"[{Path(input_path).name}] Done -> {output_path}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="datasets/smiles", show_default=True, help="Directory to save the parquet file")
@click.option("--batch-size", default=10_000, show_default=True, type=int)
def main(input: str, output_dir: str, batch_size: int) -> None:

    input_path, output_path = Path(input), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_path = output_path / f"{input_path.stem}.parquet"
    convert_graph6_to_smiles(str(input_path), str(output_path), batch_size)


if __name__ == "__main__":
    main()
