## prepare data
store data in h5 files
```bash
convert_ski10.py
```

## training
### T1 + (Equispaced) 1/4 T2 -> T2
#### GAN Only
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg GAN-Only \
--intel_stop 5e4 --coils 1 \
--prefetch
```
#### Reconstruciton after GAN (GAN)
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANRec \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 1 \
--resume /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--prefetch
```
#### Reconstruction only (Rec)
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_Rec \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 1 \
--copy_mask /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--prefetch
```
#### Baseline (ResNet)
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_None \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg None \
--intel_stop 5e4 --coils 1 \
--copy_mask /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--prefetch
```
#### Proposed (Rec-Reg)
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_CombinedMixed \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg Mixed \
--intel_stop 5e4 --coils 1 \
--copy_mask /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--resume "" \
--prefetch
```

# Evaluation
```bash
python3 eval.py \
--resume /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_CombinedMixed/ckpt/best.pth \
--save ~/save1 \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--protocals T1_POST T1 \
--aux_aug None
```
