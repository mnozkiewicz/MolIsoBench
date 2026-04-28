import importlib.util
import json
import sys
from pathlib import Path
import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.core.interfaces import EmbeddingModel


def load_wrapper(model_dir: Path):
    spec = importlib.util.spec_from_file_location("wrapper", model_dir / "wrapper.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model_dir))
    spec.loader.exec_module(module)
    return module


def embed_shard(model: EmbeddingModel, input_path: Path, output_path: Path, batch_size: int) -> None:
    df = pd.read_parquet(input_path)
    writer = None

    with tqdm(total=len(df), desc=f"{input_path.parent.name}/{input_path.name}", unit="mol") as pbar:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            embeddings = model.embed(batch["smiles"].tolist()).astype(np.float32)
            emb_dim = embeddings.shape[1]

            table = pa.table({
                "id": pa.array(batch["id"]),
                "embedding": pa.FixedSizeListArray.from_arrays(embeddings.ravel(), emb_dim),
            })

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            pbar.update(len(batch))

    if writer:
        writer.close()


@click.command()
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="datasets/embeddings", show_default=True)
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--model-kwargs", default="{}", show_default=True)
@click.option("--run-name", default=None)
def main(model_dir, input, output_dir, batch_size, model_kwargs, run_name):
    model_dir_path = Path(model_dir)
    input_path = Path(input)

    # output: <output_dir>/<run-name>/<dataset>/<shard>.parquet
    out = Path(output_dir) / (run_name or model_dir_path.name) / input_path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / input_path.name

    wrapper = load_wrapper(model_dir_path)
    model = wrapper.get_embedder(**json.loads(model_kwargs))
    embed_shard(model, input_path, output_path, batch_size)


if __name__ == "__main__":
    main()
