import argparse
import datetime
import json

from tqdm import tqdm
from lib.pipeline import RAGPipeline
from lib.preprocess import TextDataset, splitter_choices
from lib.retrieval import BM25Retriever, VectorRetriever
from lib.generate import TogetherGeneratorBase
import pandas as pd
import time
import os
from pprint import pprint
import pickle

def gt_args():
    parser = argparse.ArgumentParser(description='Run RAG system')
    parser.add_argument('--sleeptime', type=float, default=0.5)
    parser.add_argument('--in_mem_index', action="store_true", default=True)
    parser.add_argument('--lang', type=str, choices=['jv', 'kn', 'su', 'sw'], default='jv')
    parser.add_argument('--testset', type=str, choices=['langdata', 'translate-test'], default='langdata')
    parser.add_argument('--retriever', type=str, choices=['bm25', 'vec'], default='bm25')
    parser.add_argument('--generator', 
                        type=str, 
                        choices=["meta-llama/Llama-2-7b-chat-hf", 
                                 "mistralai/Mistral-7B-Instruct-v0.2", 
                                 "Qwen/Qwen1.5-7B-Chat"], 
                        default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument('--k', type=int, default=4)
    return parser.parse_args()

def mk_dataset(args):
    dataset = TextDataset(
                data_dir=f"{args.lang}-bm25-50000",
                text_splitter = splitter_choices['recursive_char_text_splitter']
            )
    dataset.print_summary()
    return dataset

def mk_retriever(args, dataset, do_index: bool):
    print(args)
    print(dataset)
    match args.retriever:
        case 'bm25':
            return BM25Retriever(dataset=dataset)
        case 'vec':
            return VectorRetriever(dataset=dataset, model_name = "intfloat/e5-large-v2")
        case _:
            raise ValueError

def mk_generator(args):
    return TogetherGeneratorBase(model_name = args.generator, api_key = "e7ca8611cd31e8d7a42c4ebec13d273ac01da59a2f853f7db00355f0368e8ba6")

def prompt_template(batch):
    q = batch["startphrase"]
    o1 = batch["ending1"]
    o2 = batch["ending2"]
    u = f''''{q}'
Which of the following corresponds to the meaning of the phrase above, enclosed in single quotes?
1: {o1}
2: {o2}
Respond with a single number: 1 or 2.'''
    return u, "\n".join([q, o1, o2])

def do_evaluation(generator: TogetherGeneratorBase, args):
    
    with open(f'../experiment/{args.lang}/retrieval_acc.pkl', 'rb') as file:
        retrieval_acc = pickle.load(file)
        
    res_acc = []
    for i, entry in tqdm(enumerate(retrieval_acc), desc="evaluating...", total=len(retrieval_acc)):
     
        (Qq, Q), A = prompt_template(entry), entry['labels']
        try_k = args.k
        retry = True
        strike = 0
        while retry:
            docs = entry['retrieved'][:min(try_k, len(entry['retrieved']))] # expect enough docs cached
            try:
                time.sleep(args.sleeptime)
                A_hat, model_output, generation_prompt = generator.answer_with_context(Q, docs)
                break
            except BaseException as e:
                print(f'failed to evaluate when k = {try_k} context window issue?, {e}')
                time.sleep(args.sleeptime)
                strike += 1
                if strike > 2:
                    try_k -= 1
                if try_k == 0:
                    A_hat, model_output, generation_prompt = 'failed', 'failed', 'failed'
                    retry = False
                    print(f'GIVE UP on entry {entry}. need to retry manually!')
                    break
                

        correct = A_hat == A

        entry_res = {
            "A_hat": A_hat,
            "correct": correct,
            **entry,
            "model_output": model_output,
            "generation_prompt": generation_prompt,
        }
        if 'Q_aug' not in entry_res:
            entry_res['Q_aug'] = '-'
        res_acc.append(entry_res)
    
    # summary stats and file IO by copilot
    output_dir = f'../experiment/{args.lang}/{args.generator.replace("/", "-")}/{args.testset}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 0 > save arguments
    args_file = f'{output_dir}/args.txt'
    with open(args_file, 'w') as f:
        f.write(str(args))
    
    # 1 > save predictions
    with open(f'{output_dir}/res_acc.pkl', 'wb') as f:
        pickle.dump(res_acc, f)

    df = pd.DataFrame(res_acc)

    df.to_csv(f'{output_dir}/predictions.csv')
    df.to_json(f'{output_dir}/predictions.json', indent=2, orient='records')
    
    # 2 > make mean calculations
    summary_stats = df[["correct"]].mean().to_frame().T.rename(columns={"correct": "Accuracy"})
    summary_stats.to_csv(f'{output_dir}/stats.csv', index=False)
    summary_stats.to_json(f'{output_dir}/stats.json', indent=2, orient='records')
    
    # 3 > bye
    print("done evaluating!")
    print("results dumped to", output_dir)
    print(summary_stats)
    
    return

def log(txt):
    with open("log.txt", "a") as o:
        o.write(f"{txt}\n")

def do_retrieval(pipeline: RAGPipeline, args):
    output_dir = f'../experiment/{args.lang}'
    if not os.path.exists(f"{output_dir}/retrieval_acc.pkl"):
        testset = pd.read_csv(f"../{args.testset}/{args.lang}.csv")
        retrieval_acc = []
        log("read testset")
        for i, entry in testset.iterrows():
            start = time.time()
            log(i)
            (Qq, Q), A = prompt_template(entry), entry['labels']
            log(Qq)
            log(Q)
            log(A)
            retrieval_res = pipeline.retrieval_pass(Q)
            retrieval_acc.append({
                **entry,
                **retrieval_res,
            })
            end = time.time() - start
            log(f"{i} {end}")
            
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f'{output_dir}/retrieval_acc.pkl', 'wb') as f:
            pickle.dump(retrieval_acc, f)
        
if __name__ == '__main__':
    args = gt_args()
    generator = mk_generator(args)
    print("made generator")
    dataset = mk_dataset(args)
    print("made dataset")
    retriever = mk_retriever(args, dataset, False)
    print("made retriever")
    pipeline = RAGPipeline(retriever, generator, args.k)
    print("made pipeline")
    do_retrieval(pipeline, args)
    do_evaluation(generator, args)
