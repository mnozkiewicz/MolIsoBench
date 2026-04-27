# MolIsoBench

## Dataset Download

Datasets are downloaded using the `download_datasets.sh` script.

By default, large `.gz` files are skipped:

```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

To download all files including large `.gz` files (which are automatically decompressed after download):

```bash
./download_datasets.sh --download-all
```

## SMILES Conversion

Convert a single graph6 file to a SMILES parquet file saved in `datasets/smiles/`:

```bash
uv run -m src.data_prep.graph6_to_smiles <input.g6> [--output-dir datasets/smiles] [--batch-size 10000]
```

Convert all graph6 files in `datasets/g6/` in parallel:

```bash
uv run prepare_all_smiles.py [--input-dir datasets/g6] [--output-dir datasets/smiles] [--workers 4] [--batch-size 10000]
```

## Embedding Models

Models live in `model_wrappers/`, each in its own subfolder with an isolated venv. Every model follows the same structure:

```
model_wrappers/{model_name}/
├── wrapper.py        # EmbeddingModel implementation + Click CLI
├── .python-version   # Python version for this model's venv
├── requirements.txt  # Model-specific dependencies
├── init.sh           # Installs requirements into the venv
└── venv/             # Created automatically on first run (gitignored)
```

Run a model via `embed_model.sh`, which creates the venv on first use and runs `wrapper.py`:

```bash
./embed_model.sh models/chembert datasets/smiles/graph8c.parquet
./embed_model.sh models/chembert datasets/smiles/graph8c.parquet --pooling mean --device cuda
./embed_model.sh models/fingerprints datasets/smiles/graph8c.parquet --n-bits 1024
```

To force reinstall the venv:

```bash
./embed_model.sh --reinstall models/chembert datasets/smiles/graph8c.parquet
```

Embeddings are saved to `datasets/embeddings/{model_name}/{dataset_name}.parquet` with columns `id` and `embedding`.

### Adding a new model

1. Create `models/{model_name}/` with the four files above
2. `wrapper.py` must implement `EmbeddingModel` from `src/core/interfaces.py` and expose `get_embedder(name, **kwargs)`
3. If the model requires a cloned repo, clone it inside the model folder and add it to `PYTHONPATH` in `init.sh`
