
```python
runs
.map((row, index) => {
    finetuned_model: row.config["finetuned_model"], 
    max_val_lang_mc_acc: row.summary["val/lang_mc_acc.max"], 
    test_lang_mc_acc: row.summary["test/mc_acc"]
})
.groupby((row) => 
    row["finetuned_model"]
)
.map((row, index) => {
    finetuned_model: row["finetuned_model"], 
    avg_max_val_lang_mc_acc: row["max_val_lang_mc_acc"].avg, 
    avg_test_lang_mc_acc: row["test_lang_mc_acc"].avg
})
```