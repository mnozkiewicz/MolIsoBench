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
