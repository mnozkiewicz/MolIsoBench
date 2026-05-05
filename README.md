# MolIsoBench

## Dataset Download

Datasets are downloaded using the `download_datasets.sh` script.
All datasets are graphs in graph6 format. There are three types of datasets: small, medium and large.

By default, only small files are downloaded.

```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

To download other you need to add appropriate flags.

```bash
./download_datasets.sh --small 
./download_datasets.sh --medium 
./download_datasets.sh --large 
./download_datasets.sh --all
```

## SMILES Conversion

Convert a single graph6 file to a SMILES parquet file saved in `datasets/smiles/`:

```bash
uv run -m src.data_prep.graph6_to_smiles <input.g6> [--output-dir datasets/smiles] [--batch-size 10000]
```

Convert all graph6 files in `datasets/g6/` in parallel:

```bash
uv run scripts/prepare_all_smiles.py [--input-dir datasets/g6] [--output-dir datasets/smiles] [--workers 4] [--shard-size 5000]
```

Convert molpile SMILES parquet (with optional atom-count filter):

```bash
uv run scripts/prepare_molpile_smiles.py <input.parquet> [--output-dir datasets/smiles] [--workers 4] [--max-atoms 50]
```

## Embedding Models

Models live in `model_wrappers/`, each in its own subfolder with an isolated venv. Every model follows the same structure:

```
model_wrappers/{model_name}/
├── wrapper.py        # EmbeddingModel implementation + get_embedder()
├── pyproject.toml    # uv-managed dependencies
├── init.sh           # Extra setup: clone repos, download weights
└── .venv/            # Created automatically by uv sync (gitignored)
```

Run a single shard via `embed_model.sh`, which syncs the venv on first use:

```bash
./embed_model.sh model_wrappers/huggingface datasets/smiles/graph6c/shard_0000.parquet --run-name chemberta-77m --model-kwargs '{"model_name":"chemberta-77m"}'
```

To force-reinstall the venv:

```bash
./embed_model.sh --reinstall model_wrappers/huggingface datasets/smiles/graph6c/shard_0000.parquet
```

Run a model across all datasets in parallel:

```bash
uv run scripts/run_embedding.py model_wrappers/huggingface --workers 4 --run-name chemberta-77m --model-kwargs '{"model_name":"chemberta-77m"}'
```

Embeddings are saved to `datasets/embeddings/{run-name}/{dataset_name}/{shard}.parquet` with columns `id` and `embedding`.

### Adding a new model

1. Create `model_wrappers/{model_name}/` with `pyproject.toml`, `wrapper.py`, and `init.sh`
2. `wrapper.py` must implement `EmbeddingModel` from `src/core/interfaces.py` and expose `get_embedder(**kwargs)`
3. If the model requires a cloned repo or downloaded weights, handle that in `init.sh`
