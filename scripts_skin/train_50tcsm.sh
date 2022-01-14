cd ..
python train_rmt_vat_mean.py --gpu 1 --out output/skin/skin50_tcsm/rmt_vat --lr 1e-4 --n-labeled 50 --consistency 1.0 --consistency_rampup 600 --epochs 800 --batch-size 16 \
--num-class 2 --val-iteration 10
