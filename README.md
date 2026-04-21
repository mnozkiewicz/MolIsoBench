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
