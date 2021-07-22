## prepare data
store data in h5 files

### fastMRI Brain DICOM set
```bash
convert_ski10.py
```

### UII private set
```bash
convert_UII_paired.py
```

## training
### T1Flair + (Equispaced) 1/8 T2 -> T2
#### GAN Only
```bash
UDA_VISIBLE_DEVICES=1 python3 train.py \
--logdir ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineGANOnly \
--train ~/UII_paired/T1Flair_T2Flair_T2_train.csv \
--val ~/UII_paired/T1Flair_T2Flair_T2_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T2 T1Flair --mask equispaced --aux_aug BSpline \
--sparsity 0.125 --epoch 20000 --batch_size 4 --reg GAN-Only \
--intel_stop 5e4 --coils 24 --prefetch
```
#### Reconstruciton after GAN
```bash
UDA_VISIBLE_DEVICES=1 python3 train.py \
--logdir ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineGANRec \
--train ~/UII_paired/T1Flair_T2Flair_T2_train.csv \
--val ~/UII_paired/T1Flair_T2Flair_T2_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T2 T1Flair --mask equispaced --aux_aug BSpline \
--sparsity 0.125 --epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 24 \
--resume ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineGANOnly/ckpt/best.pth\
--prefetch
```
#### Reconstruction only
```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py \
--logdir ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineRec \
--train ~/UII_paired/T1Flair_T2Flair_T2_train.csv \
--val ~/UII_paired/T1Flair_T2Flair_T2_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T2 T1Flair --mask equispaced --aux_aug BSpline \
--sparsity 0.125 --epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 24 \
--copy_mask ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineGANOnly/ckpt/best.pth \
--prefetch
```
#### Baseline
```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py \
--logdir ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineNone \
--train ~/UII_paired/T1Flair_T2Flair_T2_train.csv \
--val ~/UII_paired/T1Flair_T2Flair_T2_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--epoch 20000 --batch_size 4 \
--protocals T2 T1Flair --mask equispaced --aux_aug BSpline --sparsity 0.125 \
--reg None --intel_stop 5e4 --coils 24 \
--copy_mask ~/LOG/CSGAN/UII/UII_T1Flair_8xEquispacedT2_BSplineGANOnly/ckpt/best.pth \
--prefetch
```

# Evaluation
```bash
python3 eval.py --resume /mnt/workspace/LOG/CSGAN/UII_MCReg/UII_T2_8xRandomT2Flair_BSplineRec/ckpt/best.pth --save ~/save1 --val ~/UII_paired/T1Flair_T2Flair_T2_val.csv --protocals T2Flair T2 --aux_aug None
```

## training
### T1 + (Equispaced) 1/8 T2 -> T2
#### GAN Only
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.125 \
--epoch 20000 --batch_size 4 --reg GAN-Only \
--intel_stop 5e4 --coils 1 \
--prefetch
```
#### Reconstruciton after GAN
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANRec \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.125 \
--epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 1 \
--resume /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--prefetch
```
#### Reconstruction only
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_Rec \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.125 \
--epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 1 \
--copy_mask /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--prefetch
```
#### Baseline
```bash
python3 train.py \
--logdir /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_None \
--train /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_train.csv \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 \
--protocals T1_POST T1 --mask equispaced --aux_aug None --sparsity 0.125 \
--epoch 20000 --batch_size 4 --reg None \
--intel_stop 5e4 --coils 1 \
--copy_mask /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--prefetch
```

# Evaluation
```bash
python3 eval.py \
--resume /mnt/workspace/LOG/CSGAN/fastMRI_BrainDICOM/T1_4xRandomT1POST_GANOnly/ckpt/best.pth \
--save ~/save1 \
--val /mnt/workspace/fastMRI_clinical/t1_t1post_paired_6875_val.csv \
--protocals T1_POST T1 \
--aux_aug None
```

# Visualization
```bash
python3 plot_quiver.py /mnt/workspace/LOG/CSGAN/EVAL/T1_8xEquispacedT2_GANNone/19_{image,sampled,rec,aux,warped,grid}.nii 2 /mnt/workspace/Sync/T1_8xEquispacedT2_GANNone.svg
python3 plot_hist.py /mnt/workspace/LOG/CSGAN/EVAL/T1_8xRandomT2_{None,GANNone,Rec,CombinedMixed}.json
python3 plot_scatter.py /mnt/workspace/LOG/CSGAN/EVAL//UII_T1Flair_8xRandomT2_PBSpline{None,GANNone,Rec,CombinedMixed}.json
python3 plot_recon.py 4 None /mnt/workspace/LOG/CSGAN/EVAL/T1_4xRandomT2_CombinedMixed/73_image.nii /mnt/workspace/LOG/CSGAN/EVAL/T1_4xRandomT2_{None,GANNone,Rec,CombinedMixed}/73_rec.nii
python3 plot_align.py 4 /mnt/workspace/Sync/T1_4xRandomT2_align_73_4.svg /mnt/workspace/LOG/CSGAN/EVAL/T1_4xRandomT2_CombinedMixed/73_{image,aux}.nii /mnt/workspace/LOG/CSGAN/EVAL/T1_4xRandomT2_{None,GANNone,Rec,CombinedMixed}/73_{warped,grid}.nii
```
