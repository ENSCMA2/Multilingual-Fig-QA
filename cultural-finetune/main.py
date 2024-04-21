from finetune import *
from tqdm import tqdm
import torch
import math
import argparse
from accelerate.utils import set_seed

def gt_args():
    global args
    parser = argparse.ArgumentParser(description='Make dataset')
    parser.add_argument('--corpus_truncate', type=int, default=500)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--pretrained_model', type=str, choices=['FacebookAI/xlm-roberta-base'], default='FacebookAI/xlm-roberta-base')
    parser.add_argument('--corpus_chunk_size', type=int, default=128)
    parser.add_argument('--mask_probability', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mc_loss_weight', type=float, default=1.0)
    parser.add_argument('--mc_sample_weight', type=float, default=0.6)
    parser.add_argument('--mlm_loss_weight', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)    
    parser.add_argument('--cultural_corpus', type=str, choices=['su-bm25-50000'], default='su-bm25-50000')
    parser.add_argument('--resultsize', type=int, default=50000)
    parser.add_argument('--lang', type=str, choices=["hi", "id", "jv", "kn", "su", "sw", "yo"], default='su')
    parser.add_argument('--dev', action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print("args:", args)
    return args

def iter_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

def run_train_loop(
    args,
    model: MultiTaskModel, 
    corpus_train_dataloader,
    corpus_val_dataloader,
    optimizer,
    lr_scheduler,
    figqa_train_dataloader,
    figqa_val_dataloader,
    figqa_test_dataloader,
    accelerator,
    lm_corpus,
    figqa_datasets,
    interleave_probs = [0.1, 0.9],
    steps_per_epoch = 10,
):
    corpus_train_iterator = iter(corpus_train_dataloader)
    figqa_train_iterator = iter(figqa_train_dataloader)

    for epoch in tqdm(range(args.max_epochs)):
        print("\n\n==========\n🔄 EPOCH: ", epoch)
        
        # 1 > train
        model.train()
        
        # interleave: by probability, cycle through datasets, break at max interleave step
        
        for step in range(steps_per_epoch):
            interleave_index = torch.multinomial(torch.tensor(interleave_probs), 1).item()
            if interleave_index == 0:
                print('🎲 training mlm...')
                corpus_batch = iter_next(corpus_train_iterator)
                if corpus_batch == None:
                    corpus_train_iterator = iter(corpus_train_dataloader)
                    corpus_batch = iter_next(corpus_train_iterator)
                mlm_logits, mlm_loss = model.mlm_forward(corpus_batch)
                print("🎯 mlm loss: ", mlm_loss.item())
                loss = args.mlm_loss_weight * mlm_loss
            if interleave_index == 1:
                print('🎲 training mc...')
                figqa_batch = iter_next(figqa_train_iterator)
                if figqa_batch == None:
                    figqa_train_iterator = iter(figqa_train_dataloader)
                    figqa_batch = iter_next(figqa_train_iterator)
                mc_logits, mc_loss = model.mc_forward(figqa_batch)
                print("🎯 mc loss: ", mc_loss.item())
                loss = args.mc_loss_weight * mc_loss
              
            accelerator.backward(loss)  
            # loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # print('🏗️ training mlm...')
        # for corpus_batch in corpus_train_dataloader:
        #     mlm_logits, mlm_loss = model.mlm_forward(corpus_batch)
        #     print("🎯 mlm loss: ", mlm_loss.item())
        #     # accelerator.backward(mlm_loss)
        #     mlm_loss.backward()
        #     optimizer.step()
        #     lr_scheduler.step()
        #     optimizer.zero_grad()
        
        # print('🏗️ training figqa...')
        # for figqa_batch in figqa_train_dataloader:
        #     mc_logits, mc_loss = model.mc_forward(figqa_batch)
        #     print("🎯 mc loss: ", mc_loss.item())
        #     # accelerator.backward(mc_loss)
        #     mc_loss.backward()
        #     optimizer.step()
        #     lr_scheduler.step()
        #     optimizer.zero_grad()        
        
        
        # 2 > validate
        model.eval()
        print('🧪 evaluating mlm...')
        # model.fill_mask(f"The cat said {tokenizer.decode(tokenizer.mask_token_id)}.")
        mlm_losses = []
        for corpus_batch in corpus_val_dataloader:
            with torch.no_grad():
                mlm_logits, mlm_loss = model.mlm_forward(corpus_batch)
            mlm_losses.append(accelerator.gather(mlm_loss.repeat(args.batch_size)))
        mlm_losses = torch.cat(mlm_losses)
        mlm_losses = mlm_losses[: len(lm_corpus['val'])]
        try: perplexity = math.exp(torch.mean(mlm_losses))
        except OverflowError: perplexity = float("inf")
        print(f'😵‍💫 mlm val perplexity: {perplexity}')

        model.eval()
        print('🧪 evaluating figqa...')
        figqa_samples_seen = 0
        figqa_metric = evaluate.load("accuracy")
        for step, figqa_batch in enumerate(figqa_val_dataloader):
            with torch.no_grad():
                mc_logits, mc_loss = model.mc_forward(figqa_batch)
            predictions = mc_logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, figqa_batch["labels"]))
            # If we are in a multiprocess environment, we're cooked
            if accelerator.num_processes > 1: assert(False)
            figqa_metric.add_batch(predictions=predictions,references=references,)
        eval_figqa_metric = figqa_metric.compute()
        print(f'🪄 figqa val metric: {eval_figqa_metric}')
        

def run_test_loop(
    args,
    model: MultiTaskModel, 
    figqa_train_dataloader,
    figqa_val_dataloader,
    figqa_test_dataloader,
    accelerator,
):
    model.eval()
    print('🧪 testing figqa...')
    figqa_samples_seen = 0
    figqa_metric = evaluate.load("accuracy")
    for step, figqa_batch in enumerate(figqa_test_dataloader):
        with torch.no_grad():
            mc_logits, mc_loss = model.mc_forward(figqa_batch)
        predictions = mc_logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, figqa_batch["labels"]))
        # If we are in a multiprocess environment, we're cooked
        if accelerator.num_processes > 1: assert(False)
        figqa_metric.add_batch(predictions=predictions,references=references,)
    eval_figqa_metric = figqa_metric.compute()
    print(f'🪄 figqa metric: {eval_figqa_metric}')


def main(args):

    # 1 > load pretrained model
    print("⛳ 1. making tokenizer and models")
    tokenizer = mk_tokenizer(vars(args))
    mlm_model, mc_model = mk_models(vars(args))
    model = MultiTaskModel(mlm_model, mc_model, tokenizer)

    # 2 > load datasets
    print("⛳ 2. loading datasets")
    print("=== figqa data ===")
    figqa_datasets, figqa_data_collator = mk_figqa_dataset(vars(args), tokenizer, toy=False)
    figqa_train_dataloader, figqa_val_dataloader, figqa_test_dataloader = mk_figqa_dataloaders(figqa_datasets, figqa_data_collator, model.tokenizer, vars(args))

    print("\n=== cultural data ===")
    lm_corpus, corpus_mask_collator, corpus_wwm_collator = mk_corpus(vars(args), tokenizer, toy=False)
    corpus_train_dataloader, corpus_val_dataloader = mk_corpus_dataloaders(lm_corpus, corpus_mask_collator, model.tokenizer, vars(args))
    
    optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=300, # TODO this isn't right
    )
    
    # 3 > train
    print("⛳ 3. training")
    accelerator = Accelerator(cpu=False)

    model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_train_dataloader, figqa_test_dataloader, figqa_val_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_train_dataloader, figqa_test_dataloader, figqa_val_dataloader)
    
    run_train_loop(
        args,
        model, 
        corpus_train_dataloader,
        corpus_val_dataloader,
        optimizer,
        lr_scheduler,
        figqa_train_dataloader,
        figqa_val_dataloader,
        figqa_test_dataloader,
        accelerator,
        lm_corpus,
        figqa_datasets,
        interleave_probs = [1-args.mc_sample_weight, args.mc_sample_weight],
        steps_per_epoch = args.steps_per_epoch
    )
    
    # 4 > test
    print("⛳ 4. testing")
    run_test_loop(
        args,
        model, 
        figqa_train_dataloader,
        figqa_val_dataloader,
        figqa_test_dataloader,
        accelerator,
    )

if __name__ == '__main__':
    args = gt_args()
    if args.seed is not None:
        set_seed(args.seed)

    main(args)