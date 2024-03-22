python3 finetune.py \
train_gt.txt \
val_gt.txt \
--dataset-root /root/tiny-faces-pytorch-mod/data/MALF \
--epochs 10 \
--finetune_pretrained_weights checkpoint_50_best.pth \
--unfreeze_layers score_res3 \
--save-every 10