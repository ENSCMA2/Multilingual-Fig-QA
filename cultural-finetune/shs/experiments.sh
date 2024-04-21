# test wandb
python main.py --pretrained_model xlm-roberta-base --cultural_corpus jv-bm25-50000 --lang jv --num_corpus_epochs 2 --num_interleaved_epochs 20 --num_figqa_epochs 50 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --corpus_lr 1e-5 --interleave_lr 8e-6 --figqa_lr 5e-6 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 5000 --seed 4106092417 --corpus_chunk_size 128 

# may work
python main.py --num_interleaved_epochs 50 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --lr 1e-5 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 50 --seed 4106092417 --corpus_chunk_size 128

# seems to worked
python main.py --num_interleaved_epochs 50 --mc_loss_weight 3.0 --mlm_loss_weight 0.0 --mc_sample_weight 0.9 --lr 1e-5 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 50 --seed 4106092417 --corpus_chunk_size 128



# === where we stand ===
srun --cpus-per-gpu 1 --gres=gpu:8000:1 --mem=8g -t 4:00:00 --pty python main.py --tags wherewestand1 --pretrained_model xlm-roberta-base --cultural_corpus su-bm25-50000 --lang su --num_corpus_epochs 2 --num_interleaved_epochs 20 --num_figqa_epochs 60 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --corpus_lr 1e-5 --interleave_lr 8e-6 --figqa_lr 5e-6 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 1000 --seed 4106092417 --corpus_chunk_size 128 
srun --cpus-per-gpu 1 --gres=gpu:8000:1 --mem=8g -t 4:00:00 --pty python main.py --tags wherewestand1 --pretrained_model xlm-roberta-base --cultural_corpus jv-bm25-50000 --lang jv --num_corpus_epochs 2 --num_interleaved_epochs 20 --num_figqa_epochs 60 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --corpus_lr 1e-5 --interleave_lr 8e-6 --figqa_lr 5e-6 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 1000 --seed 4106092417 --corpus_chunk_size 128 
srun --cpus-per-gpu 1 --gres=gpu:8000:1 --mem=8g -t 4:00:00 --pty python main.py --tags wherewestand1 --pretrained_model xlm-roberta-base --cultural_corpus kn-bm25-50000 --lang kn --num_corpus_epochs 2 --num_interleaved_epochs 20 --num_figqa_epochs 60 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --corpus_lr 1e-5 --interleave_lr 8e-6 --figqa_lr 5e-6 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 1000 --seed 4106092417 --corpus_chunk_size 128 
srun --cpus-per-gpu 1 --gres=gpu:8000:1 --mem=8g -t 4:00:00 --pty python main.py --tags wherewestand1 --pretrained_model xlm-roberta-base --cultural_corpus sw-bm25-50000 --lang sw --num_corpus_epochs 2 --num_interleaved_epochs 20 --num_figqa_epochs 60 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --corpus_lr 1e-5 --interleave_lr 8e-6 --figqa_lr 5e-6 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 1000 --seed 4106092417 --corpus_chunk_size 128 
srun --cpus-per-gpu 1 --gres=gpu:8000:1 --mem=8g -t 4:00:00 --pty python main.py --tags wherewestand1 --pretrained_model xlm-roberta-base --cultural_corpus yo-bm25-10000 --lang yo --num_corpus_epochs 2 --num_interleaved_epochs 20 --num_figqa_epochs 60 --mc_loss_weight 3.0 --mlm_loss_weight 1.0 --mc_sample_weight 0.6 --corpus_lr 1e-5 --interleave_lr 8e-6 --figqa_lr 5e-6 --batch_size 32 --steps_per_interleaved_epoch 32 --corpus_truncate 1000 --seed 4106092417 --corpus_chunk_size 128 