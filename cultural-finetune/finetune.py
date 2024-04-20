from datasets import load_dataset
import spacy
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
from transformers import ( CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, AutoModelForMultipleChoice, AutoTokenizer, PreTrainedTokenizerBase, SchedulerType, default_data_collator, get_scheduler, AutoModel, XLMRobertaTokenizer, XLMRobertaXLModel, AutoModelForMaskedLM, XLMRobertaXLConfig, XLMRobertaXLForMultipleChoice)
from peft import AutoPeftModel
from torch.utils.data import DataLoader
import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator
import collections
from transformers import TrainingArguments
from transformers import Trainer

args = {
    'cultural_corpus': 'yo-bm25-10000',
    # 'pretrained_model': 'FacebookAI/xlm-roberta-base', 
    'pretrained_model': 'distilbert-base-uncased', 
    'corpus_chunk_size': 128,
    'wwm_probability': 0.15,
    'batch_size': 64,
}

toy_figqa_dataset = DatasetDict({
    'train': Dataset.from_dict({'label': [1, 1, 0, 0]*10, 'input': ['The cat said meow', "Cats say meow", 'The cat said woof', 'Cats generally bark']*10}),
    'val': Dataset.from_dict({'label': [1, 0]*20, 'input': ['A sound cats like to make is meow', 'A sound cats like to make is woof']*20})
})

toy_corpus = DatasetDict({
    'train': Dataset.from_dict({'score': [0.2, 0.1, 0.05]*100, 'example': ['The cat said meow', "Cats say meow", 'Tokenizers are so meow']*100}),
    # 'val': Dataset.from_dict({'score': [0.2]*20, 'example': ['A sound cats like to make is meow']*20})
})


def mk_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args['pretrained_model'])
    print(f"mask token id is {tokenizer.mask_token_id} and mask token is {tokenizer.decode(tokenizer.mask_token_id)}")
    print(f"tokenizer max len {tokenizer.model_max_length}")
    return tokenizer

def mk_models(args):
    mlm_model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")
    mc_model = AutoModelForMultipleChoice.from_pretrained("FacebookAI/xlm-roberta-base")
    mlm_model.to('cpu')
    mc_model.to('cpu')
    print(f'mlm_model num param: {mlm_model.num_parameters()}')
    print(f'mc_model num param: {mc_model.num_parameters()}')
    return mlm_model, mc_model

# based on hugging face docs
def fill_mask(args, model, tokenizer, input_text: str):
    model.to('cpu')
    toy_input = tokenizer(input_text, return_tensors="pt")
    token_logits = model(**toy_input).logits
    mask_token_index = torch.where(toy_input["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
    print(f'input: {input_text}')
    for i, token in enumerate(top_tokens):
        print(f"pred {i}: {input_text.replace(tokenizer.mask_token, f'_{tokenizer.decode([token])}_')}")

def mk_corpus(args, tokenizer):
    # corpus_unsplit = load_dataset("chaosarium/c4-cultural-extract", revision=args['cultural_corpus'])
    # corpus_unsplit['train'] = corpus_unsplit['train'].select(range(100)) # TODO just for development
    corpus_unsplit = toy_corpus
    corpus = corpus_unsplit["train"].train_test_split(test_size=0.1, seed=42)
    corpus['val'] = corpus.pop('test')

    def tokenize_function(examples): # based on hugging face docs
        result = tokenizer(examples["example"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    # tokenized_dataset = dataset.select(range(NUM_EXAMPLES)).map(tokenize_function, batched=True)
    tokenized_copus = corpus.map(tokenize_function, batched=True, remove_columns=["example", "score"])
    
    def group_texts(examples): # based on hugging face docs
        chunk_size = args['corpus_chunk_size']
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()} # Concatenate all texts
        total_length = len(concatenated_examples[list(examples.keys())[0]]) # Compute length of concatenated texts
        total_length = (total_length // chunk_size) * chunk_size # We drop the last chunk if it's smaller than chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)] 
            for k, t in concatenated_examples.items()
        } # Split by chunks of max_len
        result["labels"] = result["input_ids"].copy() # Create a new labels column
        return result
    
    lm_corpus = tokenized_copus.map(group_texts, batched=True)

    wwm_probability = args['wwm_probability']
    def whole_word_masking_data_collator(features):
        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)
    
    print("index 0 train example:")
    print(lm_corpus["train"][0])
    corpus_mask_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    corpus_wwm_collator = whole_word_masking_data_collator

    return lm_corpus, corpus_mask_collator, corpus_wwm_collator