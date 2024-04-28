from pprint import pprint
from finetune import *
from tqdm import tqdm
import torch
import math
import argparse
from accelerate.utils import set_seed
import psutil
import random
import wandb
from peft import PeftModel, PeftConfig

# TODOs
# - make real validation sets?
# - lora

def gt_args():
    global args
    parser = argparse.ArgumentParser(description='Make dataset')
    parser.add_argument('--pretrained_model', type=str, 
        choices=[
            'xlm-roberta-base', 
            'bert-base-multilingual-cased', 
            'xlm-roberta-large',
        ], 
        default='xlm-roberta-base'
    )
    parser.add_argument('--finetuned_model', type=str, 
        choices=[
            'su-1-0.005-100-4106092417-lora', 
            'su-1-0.005-10000-4106092417-lora',
            'jv-1-0.005-10000-4106092417-lora',
        ], 
        default=None
    )
    parser.add_argument('--cultural_corpus', type=str, choices=['su-bm25-50000', 'jv-bm25-50000', 'yo-bm25-10000', 'kn-bm25-50000', 'sw-bm25-50000'], default='su-bm25-50000')
    parser.add_argument('--lang', type=str, choices=["hi", "id", "jv", "kn", "su", "sw", "yo"], default='su')
    
    parser.add_argument('--corpus_truncate', type=int, default=500)
    parser.add_argument('--corpus_chunk_size', type=int, default=256)
    parser.add_argument('--mask_probability', type=float, default=0.15)
    
    parser.add_argument('--num_corpus_epochs', type=int, default=3)
    parser.add_argument('--num_interleaved_epochs', type=int, default=3)
    parser.add_argument('--num_figqa_epochs', type=int, default=3)
    parser.add_argument('--corpus_lr', type=float, default=1e-5)
    parser.add_argument('--interleave_lr', type=float, default=1e-5)
    parser.add_argument('--figqa_lr', type=float, default=1e-5)
    parser.add_argument('--steps_per_interleaved_epoch', type=int, default=30)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    
    parser.add_argument('--batch_size', type=int, default=128)    
    parser.add_argument('--mc_loss_weight', type=float, default=1.0)
    parser.add_argument('--mc_sample_weight', type=float, default=0.6)
    parser.add_argument('--mlm_loss_weight', type=float, default=1.0)
    
    parser.add_argument('--stage_corpus', action="store_true")
    
    parser.add_argument('--dev', action="store_true")
    parser.add_argument('--tags', type=str, nargs='*', default=[], help="Tags for wandb run")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    if args.cultural_corpus[:2] != args.lang:
        print("mismatched language")
        assert False
    
    print("args:", args)
    return args

def iter_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

def eval_mlm(
    run,
    model,
    corpus_val_dataloader,
    accelerator,
    lm_corpus,
    trainloopbar,
    global_epoch,
    global_step
):
    # print('🧪 valuating mlm...')
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
    # print(f'🫤 mlm val perplexity: {perplexity}')
    run.log({"val/mlm_perplexity": perplexity, "epoch": global_epoch, "step": global_step})
    run.log({"val/mlm_loss": torch.mean(mlm_losses).item(), "epoch": global_epoch, "step": global_step})
    trainloopbar.set_postfix(val_mlm_perplexity=perplexity)
    trainloopbar.set_postfix(val_mlm_loss=torch.mean(mlm_losses).item())

def eval_mc(
    run,
    model,
    figqa_eval_dataloader,
    accelerator,
    trainloopbar,
    global_epoch,
    global_step,
    prefix: str,
    write_prog_bar_val_mc_acc: bool = False
) -> None:
    figqa_samples_seen = 0
    figqa_metric = evaluate.load("accuracy")

    epochbar = tqdm(enumerate(figqa_eval_dataloader), unit="batch", leave=False, total = len(figqa_eval_dataloader))
    epochbar.set_description(f"Eval Epoch {global_epoch}")

    for step, figqa_batch in epochbar:
        with torch.no_grad():
            mc_logits, mc_loss = model.mc_forward(figqa_batch)
        predictions = mc_logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, figqa_batch["labels"]))
        # If we are in a multiprocess environment, we're cooked
        if accelerator.num_processes > 1: assert(False)
        figqa_metric.add_batch(predictions=predictions,references=references,)
    eval_figqa_metric = figqa_metric.compute()
    # print(f'🪄 figqa val metric: {eval_figqa_metric}')
    return eval_figqa_metric['accuracy']
        

# WARN: using this to train figqa is deprecated
def run_interleaved_train_loop(
    args,
    run,
    model: MultiTaskModel, 
    corpus_train_dataloader,
    corpus_val_dataloader,
    optimizer,
    lr_scheduler,
    figqa_train_dataloader,
    figqa_val_dataloader, # backward competibility: use figqa_en_val_dataloader
    figqa_test_dataloader,
    accelerator,
    lm_corpus,
    figqa_datasets,
    steps_per_epoch,
    num_epochs, 
    global_epoch: int,
    global_step: int,
    interleave_probs = [0.1, 0.9],
    gradient_accumulation_steps = 8,
):
    corpus_train_iterator = iter(corpus_train_dataloader)
    figqa_train_iterator = iter(figqa_train_dataloader)

    trainloopbar = tqdm(range(num_epochs), unit="epoch")
    trainloopbar.set_description(f"Train loop")
    for _ in trainloopbar:
        # print("\n\n==========\n🔄 EPOCH: ", global_epoch)
        
        # 1 > train
        # interleave: by probability, cycle through datasets, break at max interleave step
        model.train()
        epochbar = tqdm(range(steps_per_epoch), unit="batch", leave=False)
        epochbar.set_description(f"Epoch {global_epoch}")
        for step in epochbar:
            run.log({"train/lr": optimizer.param_groups[0]['lr'], "epoch": global_epoch, "step": global_step})
            interleave_index = torch.multinomial(torch.tensor(interleave_probs), 1).item()
            if interleave_index == 0:
                corpus_batch = iter_next(corpus_train_iterator)
                if corpus_batch == None:
                    corpus_train_iterator = iter(corpus_train_dataloader)
                    corpus_batch = iter_next(corpus_train_iterator)
                mlm_logits, mlm_loss = model.mlm_forward(corpus_batch)
                # print("\t📚 mlm loss: ", mlm_loss.item())
                # print(f'🫤 mlm val perplexity: {perplexity}')
                try: perplexity = math.exp(torch.mean(mlm_loss))
                except OverflowError: perplexity = float("inf")
                run.log({"train/mlm_perplexity": perplexity, "epoch": global_epoch, "step": global_step})
                run.log({"train/mlm_loss": mlm_loss.item(), "epoch": global_epoch, "step": global_step})
                epochbar.set_postfix(train_mlm_loss=mlm_loss.item())
                epochbar.set_postfix(train_mlm_perplexity=perplexity)
                loss = args.mlm_loss_weight * mlm_loss
            if interleave_index == 1:
                figqa_batch = iter_next(figqa_train_iterator)
                if figqa_batch == None:
                    figqa_train_iterator = iter(figqa_train_dataloader)
                    figqa_batch = iter_next(figqa_train_iterator)
                mc_logits, mc_loss = model.mc_forward(figqa_batch)
                # print("\t🗳️ mc loss: ", mc_loss.item())
                run.log({"train/mc_loss": mc_loss.item(), "epoch": global_epoch, "step": global_step})
                epochbar.set_postfix(train_mc_loss=mc_loss.item())
                loss = args.mc_loss_weight * mc_loss
            
            loss = loss / gradient_accumulation_steps
            run.log({"train/scaled_loss": loss.item(), "epoch": global_epoch, "step": global_step})
            accelerator.backward(loss)  
            
            lr_scheduler.step()
            if step % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                optimizer.step()
                optimizer.zero_grad()
            global_step+=1
            
            if step % 256 == 0:
                eval_mlm(run, model, corpus_val_dataloader, accelerator, lm_corpus, trainloopbar, global_epoch, global_step)
            
        global_epoch+=1
            
        # 2 > validate
        model.eval()
        if interleave_probs[0] == 1.0:
            eval_mlm(run, model, corpus_val_dataloader, accelerator, lm_corpus, trainloopbar, global_epoch, global_step)
        if interleave_probs[0] != 1.0 and global_epoch % 16 == 0:
            eval_mlm(run, model, corpus_val_dataloader, accelerator, lm_corpus, trainloopbar, global_epoch, global_step)
        if global_epoch % 2 == 0:
            eval_mc(run, model, figqa_val_dataloader, accelerator, trainloopbar, global_epoch, global_step, prefix='val/en')
            eval_mc(run, model, figqa_test_dataloader, accelerator, trainloopbar, global_epoch, global_step, prefix='val/lang')

def run_figqa_train_loop(
    args,
    run,
    epoch_stats,
    model: MultiTaskModel, 
    corpus_train_dataloader,
    corpus_val_dataloader,
    optimizer,
    lr_scheduler,
    figqa_en_train_dataloader,
    figqa_en_val_dataloader,
    figqa_lang_val_dataloader,
    figqa_lang_test_dataloader,
    accelerator,
    lm_corpus,
    figqa_datasets,
    num_epochs, 
    global_epoch: int,
    global_step: int,
    gradient_accumulation_steps = 8,
):

    trainloopbar = tqdm(range(num_epochs), unit="epoch")
    trainloopbar.set_description(f"Train loop")
    for _ in trainloopbar:
        # print("\n\n==========\n🔄 EPOCH: ", global_epoch)
        
        # 1 > train
        # interleave: by probability, cycle through datasets, break at max interleave step
        model.train()
        epochbar = tqdm(enumerate(figqa_en_train_dataloader), unit="batch", leave=False, total = len(figqa_en_train_dataloader))
        epochbar.set_description(f"Epoch {global_epoch}")
        for step, figqa_batch in epochbar:
            run.log({"train/lr": optimizer.param_groups[0]['lr'], "epoch": global_epoch, "step": global_step})
            
            mc_logits, mc_loss = model.mc_forward(figqa_batch)
            run.log({"train/mc_loss": mc_loss.item(), "epoch": global_epoch, "step": global_step})
            epochbar.set_postfix(train_mc_loss=mc_loss.item())
            loss = args.mc_loss_weight * mc_loss
            
            loss = loss / gradient_accumulation_steps
            run.log({"train/scaled_loss": loss.item(), "epoch": global_epoch, "step": global_step})
            accelerator.backward(loss)  
            
            lr_scheduler.step()
            if step % gradient_accumulation_steps == 0 or step == len(figqa_en_train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            global_step+=1            
            
        # 2 > validate
        model.eval()
        if global_epoch % 24 == 0:
            eval_mlm(run, model, corpus_val_dataloader, accelerator, lm_corpus, trainloopbar, global_epoch, global_step)
        
        en_val_acc = eval_mc(run, model, figqa_en_val_dataloader, accelerator, trainloopbar, global_epoch, global_step, prefix='val/en')
        run.log({f"val/en_mc_acc": en_val_acc, "epoch": global_epoch, "step": global_step})
        trainloopbar.set_postfix(val_mc_acc=en_val_acc)
        lang_val_acc = eval_mc(run, model, figqa_lang_val_dataloader, accelerator, trainloopbar, global_epoch, global_step, prefix='val/lang')
        run.log({f"val/lang_mc_acc": lang_val_acc, "epoch": global_epoch, "step": global_step})
        lang_test_acc = eval_mc(run, model, figqa_lang_test_dataloader, accelerator, trainloopbar, global_epoch, global_step, prefix='test/lang')
        run.log({f"test/lang_mc_acc": lang_test_acc, "epoch": global_epoch, "step": global_step})
        
        epoch_stats.append({
            'global_epoch': global_epoch,
            'val/en_mc_acc': en_val_acc,
            'val/lang_mc_acc': lang_val_acc,
            'test/lang_mc_acc': lang_test_acc,
        })
        
        global_epoch+=1

def main(args):

    # 1 > load pretrained model
    print("⛳ 1. making tokenizer and models")
    tokenizer = mk_tokenizer(vars(args))
    mlm_model, mc_model = mk_models(vars(args))
    model = MultiTaskModel(mlm_model, mc_model, tokenizer)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        # target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        # task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=["pooler", "mlm_head", "mc_head"],
    ))
    print(f'trainable: {model.print_trainable_parameters()}')
    if args.finetuned_model is not None:
        model = PeftModel.from_pretrained(model, 'chaosarium/cultural-finetune', revision=args.finetuned_model, is_trainable=True)
        print("❗ LOADED CULTURAL FINETUNED PARAMETERS ❗")
        print("we'll further train the following:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    # 2 > load datasets
    print("⛳ 2. loading datasets")
    print("=== figqa data ===")
    figqa_datasets, figqa_data_collator = mk_figqa_dataset(vars(args), tokenizer, toy=args.dev)
    figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader = mk_figqa_dataloaders(figqa_datasets, figqa_data_collator, model.tokenizer, vars(args))

    print("\n=== cultural data ===")
    lm_corpus, corpus_mask_collator, corpus_wwm_collator = mk_corpus(vars(args), tokenizer, toy=args.dev)
    corpus_train_dataloader, corpus_val_dataloader = mk_corpus_dataloaders(lm_corpus, corpus_mask_collator, model.tokenizer, vars(args))
    
    # 3 > setup experiment
    run = wandb.init(
        entity = 'chaosarium',
        project = 'multi', 
        config=vars(args),
        tags=['corpus-interleave-figqa', args.lang] + args.tags,
        allow_val_change=True,
    )
    global_epoch = 0
    global_step = 0
    wandb.define_metric(f'test/mc_acc')
    wandb.define_metric(f'val/en_mc_acc', summary='max')
    wandb.define_metric(f'val/lang_mc_acc', summary='max')
    wandb.define_metric(f'test/lang_mc_acc', summary='max')
    wandb.define_metric(f'val/mlm_perplexity', summary='max')
    wandb.define_metric(f'val/mlm_loss', summary='max')
    epoch_stats = []

    # 4 > train corpus only
    print("⛳ 4. cultural-extract corpus training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.corpus_lr)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=(args.num_corpus_epochs*len(corpus_train_dataloader))//15,
        num_training_steps=args.num_corpus_epochs*len(corpus_train_dataloader),
    )
    accelerator = Accelerator(cpu=False)
    model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader)
    run_interleaved_train_loop(
        args=args,
        run=run,
        model=model,
        corpus_train_dataloader=corpus_train_dataloader,
        corpus_val_dataloader=corpus_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        figqa_train_dataloader=figqa_en_train_dataloader,
        figqa_val_dataloader=figqa_en_val_dataloader,
        figqa_test_dataloader=figqa_lang_test_dataloader,
        accelerator=accelerator,
        lm_corpus=lm_corpus,
        figqa_datasets=figqa_datasets,
        steps_per_epoch=len(corpus_train_dataloader),
        num_epochs=args.num_corpus_epochs,
        global_epoch=global_epoch,
        global_step=global_step,
        interleave_probs=[1.0, 0.0],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if (args.stage_corpus):
        model.push_to_hub('chaosarium/cultural-finetune', revision=f'{args.lang}-{args.num_corpus_epochs}-{args.corpus_lr}-{args.corpus_truncate}-{args.seed}-lora')
        return

    # 5 > train interleaved
    print("⛳ 5. interleaved training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.interleave_lr)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=(args.num_interleaved_epochs*args.steps_per_interleaved_epoch)//10,
        num_training_steps=args.num_interleaved_epochs*args.steps_per_interleaved_epoch,
    )
    accelerator = Accelerator(cpu=False)
    model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader)
    run_interleaved_train_loop(
        args=args,
        run=run,
        model=model,
        corpus_train_dataloader=corpus_train_dataloader,
        corpus_val_dataloader=corpus_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        figqa_train_dataloader=figqa_en_train_dataloader,
        figqa_val_dataloader=figqa_en_val_dataloader,
        figqa_test_dataloader=figqa_lang_test_dataloader,
        accelerator=accelerator,
        lm_corpus=lm_corpus,
        figqa_datasets=figqa_datasets,
        steps_per_epoch=args.steps_per_interleaved_epoch,
        num_epochs=args.num_interleaved_epochs,
        global_epoch=global_epoch,
        global_step=global_step,
        interleave_probs=[1-args.mc_sample_weight, args.mc_sample_weight],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # 6 > train figqa only
    print("⛳ 6. figqa training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.figqa_lr)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=math.floor((args.num_figqa_epochs*len(figqa_en_train_dataloader)) * 0.15),
        num_training_steps=args.num_figqa_epochs*len(figqa_en_train_dataloader),
    )
    accelerator = Accelerator(cpu=False)
    model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, corpus_train_dataloader, corpus_val_dataloader, figqa_en_train_dataloader, figqa_en_val_dataloader, figqa_lang_val_dataloader, figqa_lang_test_dataloader)
    run_figqa_train_loop(
        args=args,
        run=run,
        epoch_stats=epoch_stats,
        model=model,
        corpus_train_dataloader=corpus_train_dataloader,
        corpus_val_dataloader=corpus_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        figqa_en_train_dataloader=figqa_en_train_dataloader,
        figqa_en_val_dataloader=figqa_en_val_dataloader,
        figqa_lang_val_dataloader=figqa_lang_val_dataloader,
        figqa_lang_test_dataloader=figqa_lang_test_dataloader,
        accelerator=accelerator,
        lm_corpus=lm_corpus,
        figqa_datasets=figqa_datasets,
        num_epochs=args.num_figqa_epochs,
        global_epoch=global_epoch,
        global_step=global_step,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 7 > grab test acc for epoch with best val acc (TODO this is really a hack without blowing up disk with checkpoints)
    print("⛳ 7. extracting test result")
    epoch_stats_sorted = sorted(epoch_stats, key=lambda entry: entry['val/lang_mc_acc'] + 0.5*entry['val/en_mc_acc'], reverse=True)
    run.log({"test/mc_acc": epoch_stats_sorted[0]['test/lang_mc_acc'], "epoch": epoch_stats_sorted[0]['global_epoch']})


if __name__ == '__main__':
    args = gt_args()
    if args.seed is not None:
        set_seed(args.seed)
    else:
        seed = random.randint(0, 4294967295)
        set_seed(seed)
        args.seed = seed
        print(f"using seed {seed}")

    main(args)