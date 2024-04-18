from datasets import load_dataset
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

args = None

def tokenize(text: str, lang: str):
    return nltk.word_tokenize(text, language=lang, preserve_line=True)

def bm25_score_docs(tokenized_queries: list[list[str]], tokenized_example_pool: list[list[str]]):
    scores_acc = np.zeros(len(tokenized_example_pool))
    bm25 = BM25Okapi(tokenized_example_pool)
    for i, tokenized_query in tqdm(enumerate(tokenized_queries), total=len(tokenized_queries)):
        scores_acc += bm25.get_scores(tokenized_query)
    scores_acc /= len(tokenized_example_pool)
    return scores_acc

class TokenizerForRouge:
    def __init__(self):
        pass
    def tokenize(self, already_tokenized):
        # return tokenize(text, args.lang)
        return already_tokenized

def rouge_score_docs(tokenized_queries: list[list[str]], tokenized_example_pool: list[list[str]]):
    scores_acc = np.zeros(len(tokenized_example_pool))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=TokenizerForRouge())
    
    for i, tokenized_query in tqdm(enumerate(tokenized_queries), total=len(tokenized_queries)):
        for j, tokenized_example in enumerate(tokenized_example_pool):              
            rouge_out = scorer.score(tokenized_query, tokenized_example)
            (rouge1, rouge2, rougel) = (rouge_out["rouge1"].fmeasure, rouge_out["rouge2"].fmeasure, rouge_out["rougeL"].fmeasure)
            rouge_overall = sum([rouge1, rouge2, rougel])
            scores_acc[j] += rouge_overall

    scores_acc /= len(tokenized_example_pool)
    return scores_acc


def gt_args():
    global args
    parser = argparse.ArgumentParser(description='Make dataset')
    parser.add_argument('--scorer', type=str, choices=['bm25', 'rouge'], default='bm25')
    parser.add_argument('--resultsize', type=int, default=50000)
    parser.add_argument('--lang', type=str, choices=["hi", "id", "jv", "kn", "su", "sw", "yo"], default='su')
    parser.add_argument('--dev', action="store_true")
    args = parser.parse_args()
    print("args:", args)

def mk_dataset():
    # load c4

    print('🕰️ loading c4...')
    c4 = load_dataset("allenai/c4", args.lang)
    # c4_dev_dict = c4.data['validation'].to_pydict()
    c4_train_dict = c4.data['train'].to_pydict()

    # load mabl

    print('🕰️ loading mabl...')
    mabl_df = pd.read_csv(f'../langdata/{args.lang}.csv')
    start_phrases = mabl_df['startphrase'].tolist()
    ending1s = mabl_df['ending1'].tolist()
    ending2s = mabl_df['ending2'].tolist()
    figqa_examples = list(map(lambda tri: f"{tri[0]} {tri[1]} {tri[2]}".replace(".", "").replace(",", ""), zip(start_phrases, ending1s, ending2s)))

    # make query and pool

    if args.dev:
        print("WARN: development mode")
        queries = figqa_examples[:100]
        example_pool = c4_train_dict['text'][:1000]
    else:
        queries = figqa_examples
        example_pool = c4_train_dict['text']
        
    print('🕰️ tokenizing...')
    tokenized_queries = [tokenize(text, args.lang) for text in tqdm(queries)]
    tokenized_example_pool = [tokenize(text, args.lang) for text in tqdm(example_pool)]
    
    print('🕰️ scoring...')
    match args.scorer:
        case 'bm25':
            score_arr = bm25_score_docs(tokenized_queries, tokenized_example_pool)
        case 'rouge':
            score_arr = rouge_score_docs(tokenized_queries, tokenized_example_pool)
        case _ :
            raise ValueError
    
    sorted_scored_examples = []
    for score, example in zip(score_arr, example_pool):
        sorted_scored_examples.append({'score': score, 'example': example})
    sorted_scored_examples.sort(key=lambda x: -x['score'])
    
    scores_list = list(map(lambda x: x['score'], sorted_scored_examples[:args.resultsize]))
    example_list = list(map(lambda x: x['example'], sorted_scored_examples[:args.resultsize]))
    curated_hf_dataset = Dataset.from_dict({'score': scores_list, 'example': example_list})
    
    dev_indicator = 'dev/' if args.dev else ''
    
    curated_hf_dataset.save_to_disk(f'select_datasets/{dev_indicator}{args.lang}/{args.scorer}-{args.resultsize}')
    
    sns.histplot(scores_list)
    plt.title("score distribution among selected examples")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.savefig(f'select_datasets/{dev_indicator}{args.lang}/{args.scorer}-{args.resultsize}/score_dist.png')

if __name__ == '__main__':
    gt_args()
    mk_dataset()