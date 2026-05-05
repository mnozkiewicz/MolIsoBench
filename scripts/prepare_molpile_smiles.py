from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from tqdm import tqdm

_SCHEMA = pa.schema([("id", pa.int64()), ("smiles", pa.string())])


def _process_row_groups(
    worker_id: int,
    input_path: str,
    row_group_indices: list[int],
    row_offsets: list[int],
    output_dir: str,
    shard_size: int,
    max_atoms: int,
) -> tuple[int, int]:
    
    RDLogger.DisableLog("rdApp.*")

    pf = pq.ParquetFile(input_path)
    buffers: dict[int, list] = defaultdict(list)
    shard_counters: dict[int, int] = defaultdict(int)
    skipped = 0

    def flush(n_atoms: int) -> None:
        out_dir = Path(output_dir) / f"molpile_atoms_{n_atoms}"
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(
            pd.DataFrame(buffers[n_atoms]), schema=_SCHEMA, preserve_index=False
        )
        pq.write_table(table, out_dir / f"w{worker_id:03d}_s{shard_counters[n_atoms]:04d}.parquet")
        shard_counters[n_atoms] += 1
        buffers[n_atoms].clear()

    total_rows = sum(
        pf.metadata.row_group(rg_idx).num_rows for rg_idx in row_group_indices
    )
    with tqdm(total=total_rows, desc=f"worker {worker_id}", position=worker_id, leave=True, unit="mol") as pbar:
        for rg_idx, row_offset in zip(row_group_indices, row_offsets):
            smiles_col = pf.read_row_group(rg_idx, columns=["SMILES"]).column("SMILES").to_pylist()
            for local_i, smi in enumerate(smiles_col):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    skipped += 1
                    pbar.update(1)
                    continue
                n_atoms = mol.GetNumAtoms()

                if n_atoms > max_atoms:
                    skipped += 1
                    pbar.update(1)
                    continue
                
                buffers[n_atoms].append({"id": row_offset + local_i, "smiles": smi})
                if len(buffers[n_atoms]) >= shard_size:
                    flush(n_atoms)
                pbar.update(1)

    for n_atoms in list(buffers):
        if buffers[n_atoms]:
            flush(n_atoms)

    print(f"[worker {worker_id}] done (skipped {skipped})")
    return worker_id, skipped


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="datasets/smiles", show_default=True)
@click.option("--workers", default=4, show_default=True, type=int)
@click.option("--shard-size", default=5_000, show_default=True, type=int)
@click.option("--max-atoms", default=50, show_default=True, type=int)
def main(input: str, output_dir: str, workers: int, shard_size: int, max_atoms: int) -> None:
    pf = pq.ParquetFile(input)
    metadata = pf.metadata
    n_rg = metadata.num_row_groups

    # Compute row offset for each row group
    offsets, offset = [], 0
    for i in range(n_rg):
        offsets.append(offset)
        offset += metadata.row_group(i).num_rows

    # Round-robin distribution of row groups across workers
    rg_chunks = [[] for _ in range(workers)]
    off_chunks = [[] for _ in range(workers)]
    for i in range(n_rg):
        w = i % workers
        rg_chunks[w].append(i)
        off_chunks[w].append(offsets[i])

    click.echo(f"Processing {n_rg} row groups across {workers} workers...")

    total_skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_row_groups, w, input, rg_chunks[w], off_chunks[w], output_dir, shard_size, max_atoms)
            for w in range(workers)
            if rg_chunks[w]
        ]
        for future in as_completed(futures):
            _, skipped = future.result()
            total_skipped += skipped


if __name__ == "__main__":
    main()
