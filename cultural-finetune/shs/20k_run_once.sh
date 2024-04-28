# figqa finetue experiments
## su 20k
python main.py --tags report1 --pretrained_model xlm-roberta-base --cultural_corpus su-bm25-50000 --lang su --num_corpus_epochs 0 --num_interleaved_epochs 0 --num_figqa_epochs 120 --mc_loss_weight 1.0 --figqa_lr 5e-4 --batch_size 64 --steps_per_interleaved_epoch 32  --corpus_truncate 500 --corpus_chunk_size 128 --gradient_accumulation_steps 8 --finetuned_model su-1-0.003-20000-4106092417-lora
## jv 20k
python main.py --tags report1 --pretrained_model xlm-roberta-base --cultural_corpus jv-bm25-50000 --lang jv --num_corpus_epochs 0 --num_interleaved_epochs 0 --num_figqa_epochs 120 --mc_loss_weight 1.0 --figqa_lr 5e-4 --batch_size 64 --steps_per_interleaved_epoch 32  --corpus_truncate 500 --corpus_chunk_size 128 --gradient_accumulation_steps 8 --finetuned_model jv-1-0.003-20000-4106092417-lora
## kn 20k
python main.py --tags report1 --pretrained_model xlm-roberta-base --cultural_corpus kn-bm25-50000 --lang kn --num_corpus_epochs 0 --num_interleaved_epochs 0 --num_figqa_epochs 120 --mc_loss_weight 1.0 --figqa_lr 5e-4 --batch_size 64 --steps_per_interleaved_epoch 32  --corpus_truncate 500 --corpus_chunk_size 128 --gradient_accumulation_steps 8 --finetuned_model kn-1-0.003-20000-4106092417-lora
## sw 20k
python main.py --tags report1 --pretrained_model xlm-roberta-base --cultural_corpus sw-bm25-50000 --lang sw --num_corpus_epochs 0 --num_interleaved_epochs 0 --num_figqa_epochs 120 --mc_loss_weight 1.0 --figqa_lr 5e-4 --batch_size 64 --steps_per_interleaved_epoch 32  --corpus_truncate 500 --corpus_chunk_size 128 --gradient_accumulation_steps 8 --finetuned_model sw-1-0.003-20000-4106092417-lora
