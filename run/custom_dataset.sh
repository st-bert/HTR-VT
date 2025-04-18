python3 train.py --exp-name custom_dataset \
--max-lr 1e-4 \
--train-bs 32 \
--val-bs 8 \
--weight-decay 0.5 \
--mask-ratio 0.4 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 256 256 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
--eval-iter 100 \
--use-early-stopping \
--early-stop-patience 5 \
--use-wandb \
--max-seq-length 10 \
CUSTOM

python3 test.py --exp-name custom_dataset \
--max-lr 1e-4 \
--train-bs 32 \
--val-bs 8 \
--weight-decay 0.5 \
--mask-ratio 0.4 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 256 256 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
--max-seq-length 10 \
CUSTOM