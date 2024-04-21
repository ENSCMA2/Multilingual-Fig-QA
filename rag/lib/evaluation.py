from collections import Counter
import pandas as pd
import pickle

def load_annotated_qas(qa_jsons_paths: list[str], is_augmented):
    if is_augmented:
        qa_jsons_paths = list(map(lambda s: s.replace(".json", "_aug.json"), qa_jsons_paths))
    df_acc = []
    for filepath in qa_jsons_paths:
        df = pd.read_json(filepath, orient='records')
        df['filepath'] = filepath
        df_acc.append(df)
    
    dfs = pd.concat(df_acc).reset_index()
    return dfs.to_dict(orient='records')