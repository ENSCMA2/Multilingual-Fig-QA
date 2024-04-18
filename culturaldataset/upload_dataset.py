from datasets import load_from_disk
LANG_CODE = 'kn'
DSETSIZE = 50000
SCORER = 'bm25'
hf_dataset = load_from_disk(f"select_datasets/{LANG_CODE}/{SCORER}-{DSETSIZE}")

hf_dataset.push_to_hub(
    f"chaosarium/c4-cultural-extract",
    revision=f'{LANG_CODE}-{SCORER}-{DSETSIZE}'
)