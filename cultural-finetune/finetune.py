from datasets import load_dataset
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
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
from accelerate import Accelerator
# accelerator = Accelerator(cpu=True)
from itertools import chain
from dataclasses import dataclass
from transformers.utils import PaddingStrategy, get_full_repo_name
import random
from tqdm import tqdm
import os, json
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.init as init



toy_figqa_dataset = DatasetDict({
    'train': Dataset.from_dict({
        'labels': [0,0,1,1]*20,
        'startphrase': ['The cat said meow', "Cats say meow", 'The cat said woof', 'Cats generally bark']*20,
        'ending1': ['meow', "meow", 'meow', 'meow']*20,
        'ending2': ['woof', "woof", 'woof', 'woof']*20,
    }),
    # 'val': Dataset.from_dict({'labels': [1, 0]*20, 'input': ['A sound cats like to make is meow', 'A sound cats like to make is woof']*20})
    'val': Dataset.from_dict({
        'labels': [0,0,1,1], 
        'startphrase': ['The cat said meow', "Cats say meow", 'The cat said woof', 'Cats generally bark'],
        'ending1': ['meow', "meow", 'meow', 'meow'],
        'ending2': ['woof', "woof", 'woof', 'woof'],
    }),
    'test': Dataset.from_dict({
        'labels': [0,1], 
        'startphrase': ['A sound cats like to make is meow', "A sound cats like to make is woof"],
        'ending1': ['meow', "meow"],
        'ending2': ['woof', "woof"],
    })
    # Dataset.from_dict({'label': [1, 0]*20, 'input': ['A sound cats like to make is meow', 'A sound cats like to make is woof']*20})
})

toy_corpus = DatasetDict({
    'train': Dataset.from_dict({'score': [0.2, 0.1, 0.05]*140, 'example': ['The cat said meow', "Cats say meow", 'Tokenizers are so meow']*140}),
    # 'val': Dataset.from_dict({'score': [0.2]*20, 'example': ['A sound cats like to make is meow']*20})
})

def mk_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args['pretrained_model'])
    print(f"mask token id is {tokenizer.mask_token_id} and mask token is {tokenizer.decode(tokenizer.mask_token_id)}")
    print(f"tokenizer max len {tokenizer.model_max_length}")
    return tokenizer

def mk_models(args):
    config = AutoConfig.from_pretrained(args['pretrained_model'])
    config.output_hidden_states = True
    print(f"model config: {config}")
    mlm_model = AutoModelForMaskedLM.from_pretrained(args['pretrained_model'], config=config)
    mc_model = AutoModelForMultipleChoice.from_pretrained(args['pretrained_model'], config=config)
    mlm_model.to('cpu')
    mc_model.to('cpu')
    print(f'mlm_model num param: {mlm_model.num_parameters()}')
    print(f'mc_model num param: {mc_model.num_parameters()}')
    return mlm_model, mc_model

class MultiTaskModel(torch.nn.Module):
    def __init__(self, mlm_model, mc_model, tokenizer): # by claude
        super().__init__()
        self.mc_model = mc_model
        self.base_model = mc_model.base_model
        self.mlm_head = mlm_model.lm_head
        self.mc_head = mc_model.classifier
        # self.mc_head = torch.nn.Sequential(
        #     torch.nn.Linear(mc_model.config.hidden_size, mc_model.config.hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(mc_model.config.hidden_size, 1)
        # )
        # for p in self.mc_head.parameters():
        #     if p.dim() > 1:
        #         init.xavier_uniform_(p)
                
        self.mc_dropout = mc_model.dropout
        self.tokenizer = tokenizer
        mlm_model.base_model = self.base_model

    def mlm_forward(self, batch):       
        base_output = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        last_hidden = base_output.last_hidden_state

        mlm_logits = self.mlm_head(last_hidden)
        mlm_loss = torch.nn.functional.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), batch['labels'].view(-1))
        return mlm_logits, mlm_loss
    
    def mlm_inference(self, batch):       
        base_output = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        last_hidden = base_output.last_hidden_state
        mlm_logits = self.mlm_head(last_hidden)
        return mlm_logits

    def fill_mask(self, input_text: str):
        toy_input = self.tokenizer(input_text, return_tensors="pt").to(self.base_model.device)
        token_logits = self.mlm_inference(toy_input)
        mask_token_index = torch.where(toy_input["input_ids"] == self.tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
        print(f'input: {input_text}')
        for i, token in enumerate(top_tokens):
            print(f"pred {i}: {input_text.replace(self.tokenizer.mask_token, f'_{self.tokenizer.decode([token])}_')}")

    def mc_forward(self, batch):
        input_ids=batch["input_ids"] # N, 2, L
        attention_mask=batch["attention_mask"]
        
        num_choices = input_ids.shape[1]
        assert (num_choices == 2)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) # 2N, L
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) # 2N, L
        base_output = self.base_model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )
        
        pooled_output = base_output[1] # 2N, H
        pooled_output = self.mc_dropout(pooled_output) # 2N, H
        mc_logits = self.mc_head(pooled_output) # 2N, 1
        # print(mc_logits)
        reshaped_mc_logits = mc_logits.view(-1, num_choices) # N, 2
        # print(reshaped_mc_logits)
        # print(batch['labels'])
        
        # loss_fct = CrossEntropyLoss()
        # mc_loss = loss_fct(reshaped_mc_logits, batch["labels"])
        # mc_logits = self.mc_head(last_hidden[:, 0, :])  # Use the [CLS] token for multiple choice
        mc_loss = torch.nn.functional.cross_entropy(reshaped_mc_logits, batch["labels"])
        return reshaped_mc_logits, mc_loss

    # def forward(self, input_ids, attention_mask, mlm_labels, multiple_choice_labels): # by claude
    #     # Pass the input through the base model
    #     output = self.base_model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask
    #     )
    #     sequence_output = output.last_hidden_state

    #     # Compute the Masked Language Modeling (MLM) loss
    #     mlm_output = self.mlm_head(sequence_output)
    #     mlm_loss = torch.nn.functional.cross_entropy(mlm_output.view(-1, mlm_output.size(-1)), mlm_labels.view(-1))

    #     # Compute the Multiple Choice loss
    #     mc_output = self.mc_head(sequence_output[:, 0, :])  # Use the [CLS] token for multiple choice
    #     mc_loss = torch.nn.functional.cross_entropy(mc_output, multiple_choice_labels)

    #     # Combine the losses
    #     total_loss = mlm_loss + mc_loss
    #     return total_loss

def mk_multitask_model(args):
    mlm_model, mc_model = mk_models(args)
    return MultiTaskModel(mlm_model, mc_model)


# based on hugging face docs
def fill_mask(args, model, tokenizer, input_text: str):
    toy_input = tokenizer(input_text, return_tensors="pt")
    token_logits = model(**toy_input).logits
    mask_token_index = torch.where(toy_input["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
    print(f'input: {input_text}')
    for i, token in enumerate(top_tokens):
        print(f"pred {i}: {input_text.replace(tokenizer.mask_token, f'_{tokenizer.decode([token])}_')}")

def mk_corpus(args, tokenizer, toy: bool = True):
    if not toy:
        corpus_unsplit = load_dataset("chaosarium/c4-cultural-extract", revision=args['cultural_corpus'])
        corpus_unsplit["train"] = corpus_unsplit["train"].select(range(args['corpus_truncate']))
    else:
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
            mask = np.random.binomial(1, args['mask_probability'], (len(mapping),))
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
    print('decoded:', tokenizer.decode(lm_corpus['train'][0]['input_ids']))
    corpus_mask_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args['mask_probability'])
    corpus_wwm_collator = whole_word_masking_data_collator

    return lm_corpus, corpus_mask_collator, corpus_wwm_collator

# lm_corpus = DatasetDict({
#     train: Dataset({
#         features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
#         num_rows: 14427
#     })
#     val: Dataset({
#         features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
#         num_rows: 1434
#     })
# })
def mk_corpus_dataloaders(lm_corpus: DatasetDict, corpus_mask_collator, tokenizer, args):
    def insert_random_mask(batch): # from pytorch docs
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = corpus_mask_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
    
    no_word_ids_lm_corpus = lm_corpus.remove_columns(["word_ids"])
    val_dataset = no_word_ids_lm_corpus["val"].map(
        insert_random_mask,
        batched=True,
        remove_columns=no_word_ids_lm_corpus["val"].column_names,
    )
    val_dataset = val_dataset.rename_columns({
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    })
    print('eval instance:', val_dataset[0])
    print('eval instance:', tokenizer.decode(val_dataset[0]['input_ids']))
    print('eval instance:', ([tokenizer.decode([id]) if id >= 0 else 0 for id in val_dataset[0]['labels']]))
    print()
    print('train instance (before collate):', no_word_ids_lm_corpus["train"][0])
    print('train instance (before collate):', tokenizer.decode(no_word_ids_lm_corpus["train"][0]['input_ids']))
    print('train instance (before collate):', ([tokenizer.decode([id]) if id >= 0 else 0 for id in no_word_ids_lm_corpus["train"][0]['labels']]))
    
    train_dataloader = DataLoader(
        no_word_ids_lm_corpus["train"],
        shuffle=True,
        batch_size=args['batch_size'],
        collate_fn=corpus_mask_collator,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], collate_fn=default_data_collator)
    return train_dataloader, val_dataloader

def mk_figqa_dataloaders(figqa_datasets, figqa_data_collator, tokenizer, args):
    figqa_train_dataloader = DataLoader(figqa_datasets["train"], shuffle=True, collate_fn=figqa_data_collator, batch_size=args['batch_size'])
    figqa_val_dataloader = DataLoader(figqa_datasets["val"], collate_fn=figqa_data_collator, batch_size=args['batch_size'])
    figqa_test_dataloader = DataLoader(figqa_datasets["test"], collate_fn=figqa_data_collator, batch_size=args['batch_size'])
    for batch in figqa_train_dataloader:
        print("figqa train instance:", batch['input_ids'][0], tokenizer.decode(batch['input_ids'][0][0]))
        print("its label:", batch['labels'][0])
        break
    return figqa_train_dataloader, figqa_val_dataloader, figqa_test_dataloader

@dataclass
class DataCollatorForMultipleChoice: # from multilingual figqa
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding = True # Union[bool, str, PaddingStrategy]
    max_length = None # Optional[int]
    pad_to_multiple_of = None # Optional[int]

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def mk_figqa_dataset(args, tokenizer, toy: bool = True):
    if not toy:
        data_files = {
            'train': '../langdata/en_train.csv',
            'val': '../langdata/en_dev.csv',
            'test': f'../langdata/{args["lang"]}.csv',
        }
        raw_datasets = load_dataset('csv', data_files=data_files)
    else:
        raw_datasets = toy_figqa_dataset
    
    def preprocess_function(examples):
        column_names = ['startphrase', 'ending1', 'ending2', 'labels']
        ending_names = [f"ending{i}" for i in [1, 2]]
        context_name = "startphrase"
        label_column_name = "label" if "label" in column_names else "labels"

        first_sentences = [[context] * 2 for context in examples[context_name]]
        second_sentences = [[examples[end][i] for end in ending_names] for i in range(len(examples[context_name]))]
        labels = examples[label_column_name]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=128,
            padding=False,
            truncation=True,
        )

        # Save the decoded sentences if storing embeddings
        # if args["do_predict"] and args["save_embeddings"]:
        #     sentence_fp = os.path.join(args["embedding_output_dir"], "sentences.tsv")
        #     with open(sentence_fp, "a") as f:
        #         for i in range(len(tokenized_examples["input_ids"])):
        #             f.write(tokenizer.decode(tokenized_examples["input_ids"][i]) + "\n")

        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels

        return tokenized_inputs
    
    figqa_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names
    )
    figqa_datasets
    
    print('figqa data looks like:')
    print('train:', figqa_datasets['train'][0])
    print('decoded:', list(map(tokenizer.decode, figqa_datasets['train'][0]['input_ids'])))
    print('val:', figqa_datasets['val'][0])
    print('decoded:', list(map(tokenizer.decode, figqa_datasets['val'][0]['input_ids'])))
    print('test:', figqa_datasets['test'][0])
    print('decoded:', list(map(tokenizer.decode, figqa_datasets['test'][0]['input_ids'])))
    
    figqa_data_collator = DataCollatorForMultipleChoice(
        tokenizer, 
        # pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )
    
    return figqa_datasets, figqa_data_collator

# runs one epoch of training
def train_model(train_dataloader, model, accelerator, optimizer, lr_scheduler, args, completed_steps, checkpointing_steps, eval_dataloader=None):
    model.train()
    total_loss = 0
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # We need to skip steps until we reach the resumed step
        # if args["resume_from_checkpoint"] and epoch == starting_epoch:
        #     if resume_step is not None and step < resume_step:
        #         completed_steps += 1
        #         continue
        outputs = model(input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
        loss = outputs.loss
        # We keep track of the loss at each epoch
        total_loss += loss.detach().float()
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps }"
                if args["output_dir"] is not None:
                    output_dir = os.path.join(args["output_dir"], output_dir)
                accelerator.save_state(output_dir)

        # if completed_steps >= args["max_train_steps"]:
        #     break
        
        print(loss)

    return model, loss, completed_steps

# evaluate mc on dataloader
def eval_model(model, eval_dataloader, metric, accelerator, epoch, args):
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    accelerator.print(f"epoch {epoch}: {eval_metric}")

    return eval_metric