# figqa finetue experiments
## sw 10k
python main.py --tags report1 --pretrained_model xlm-roberta-base --cultural_corpus sw-bm25-50000 --lang sw --num_corpus_epochs 0 --num_interleaved_epochs 0 --num_figqa_epochs 120 --mc_loss_weight 1.0 --figqa_lr 5e-4 --batch_size 64 --steps_per_interleaved_epoch 32  --corpus_truncate 500 --corpus_chunk_size 128 --gradient_accumulation_steps 8 --finetuned_model sw-1-0.005-10000-4106092417-lora

## sw 10k
python main.py --tags report1 --pretrained_model xlm-roberta-base --cultural_corpus sw-bm25-50000 --lang sw --num_corpus_epochs 0 --num_interleaved_epochs 0 --num_figqa_epochs 120 --mc_loss_weight 1.0 --figqa_lr 5e-4 --batch_size 64 --steps_per_interleaved_epoch 32  --corpus_truncate 500 --corpus_chunk_size 128 --gradient_accumulation_steps 8 --finetuned_model sw-1-0.005-10000-4106092417-lora
