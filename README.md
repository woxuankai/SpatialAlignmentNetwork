# Multi-Modal MRI Reconstruction Assisted with Spatial Alignment Network
## Abstract
In clinical practice, magnetic resonance imaging (MRI) with multiple contrasts
is usually acquired in a single study
to assess different properties of the same region of interest in human body.
The whole acquisition process can be accelerated
by having one or more modalities under-sampled in the $k$-space. 
Recent researches demonstrate that,
considering the redundancy between different contrasts or modalities,
a target MRI modality under-sampled in the $k$-space can be more efficiently reconstructed
with a fully-sampled MRI contrast as the reference modality.
However, we find that the performance of the above multi-modal reconstruction
can be negatively affected by subtle spatial misalignment between different contrasts,
which is actually common in clinical practice.
In this paper, to compensate for such spatial misalignment,
we integrate the spatial alignment network with multi-modal reconstruction
towards better reconstruction quality of the target modality.
First, the spatial alignment network estimates the spatial misalignment
between the fully-sampled reference and the under-sampled target images,
and warps the reference image accordingly.
Then, the aligned fully-sampled reference image joins the multi-modal reconstruction
of the under-sampled target image.
Also, considering the contrast difference between the target and the reference images,
we particularly design the cross-modality-synthesis-based registration loss,
in combination with the reconstruction loss,
to jointly train the spatial alignment network and the reconstruction network.
Experiments on both clinical MRI and multi-coil $k$-space raw data
demonstrate the superiority and robustness of
multi-modal MRI reconstruction empowered with our spatial alignment network.
Our code is publicly available at [https://github.com/woxuankai/SpatialAlignmentNetwork](https://github.com/woxuankai/SpatialAlignmentNetwork).

## Overview
![Overview](asserts/overview.png)
The above figure is a real case demonstrating
the existence of spatial misalignment (a), and the overview of the proposed method (b).
In (a), a real case of multi-modal
MRI acquired for the diagnostic purpose demonstrates the existence
of spatial misalignment (highlighted by arrows) between the reference
(T1-weighted) and the target (T2-weighted) images.
The aligned reference image is also available to show
the effect of our proposed spatial alignment network.
In (b), a spatial alignment network is integrated into the multi-modal MRI reconstruction pipeline
to compensate for the spatial misalignment between the fully-sampled
reference image and the under-sampled target. The data flow for the
conventional deep-learning-based reconstruction is shown in black
arrows; and the red arrows are for additional data flow related to
our proposed spatial alignment network.

For more details on the proposed method, please refer to [arXiv preprint arXiv:2108.05603](https://arxiv.org/abs/2108.05603).

## Experiments on fastMRI DICOM
### Prepare data
Store data in h5 files
1. Unzip fastMRI brain DICOM to `fastMRI_brain_DICOM` folder.
2. Convert all dicom to `brain_nii` folder.
```bash
ls fastMRI_brain_DICOM | while read X; \
do XX="brain_nii/${X}"; mkdir ${XX}; \
echo "dcm2niix -z n -f '%j-%p' -o ${XX} brain_dicom/${X} 2>${XX}/error.log 1>${XX}/out.log"; \
done | parallel --bar
```
3. Convert selected nii to h5.
```bash
# T1 modality
cat t1_t2_paired_6875_train.csv t1_t2_paired_6875_val.csv | cut -f1 -d ',' | while read x; \
do python3 convert_fastMRIDICOM.py "${x%.h5}.nii" "${x}" T1; \
done
# T2 modality
cat t1_t2_paired_6875_train.csv t1_t2_paired_6875_val.csv | cut -f2 -d ',' | while read x; \
do python3 convert_fastMRIDICOM.py "${x%.h5}.nii" "${x}" T2;
done
```

### training
T1 + (Random) 1/4 T2 -> T2
#### Baseline (ResNet)
```bash
python3 train.py \
--logdir T1_4xRandomT2_None \
--train t1_t2_paired_6875_train.csv --val t1_t2_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 --smooth_weight 100 \
--protocals T2 T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg None \
--intel_stop 5e4 --coils 1 \
--copy_mask T1_4xRandomT2_GANOnly/ckpt/best.pth #--prefetch
```

#### GAN only for spatial alignment network (GAN)
First optimize Spatial Alignment Network with GAN/smoothness only.
```bash
python3 train.py \
--logdir T1_4xRandomT2_GANOnly \
--train t1_t2_paired_6875_train.csv --val t1_t2_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 --smooth_weight 100 \
--protocals T2 T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg GAN-Only \
--intel_stop 5e4 --coils 1 
#--prefetch
```
Then solely optimize Reconstruction Network with fixed Spatial Alignment Network.
```bash
python3 train.py \
--logdir T1_4xRandomT2_GANRec \
--train t1_t2_paired_6875_train.csv --val t1_t2_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 --smooth_weight 100 \
--protocals T2 T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 1 \
--resume T1_4xRandomT2_GANOnly/ckpt/best.pth #--prefetch
```
#### Reconstruction only (Rec)
```bash
python3 train.py \
--logdir T1_4xRandomT2_Rec \
--train t1_t2_paired_6875_train.csv --val t1_t2_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 --smooth_weight 100 \
--protocals T2 T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg Rec \
--intel_stop 5e4 --coils 1 \
--copy_mask T1_4xRandomT2_GANOnly/ckpt/best.pth #--prefetch
```

#### Proposed (Rec-Reg)
```bash
python3 train.py \
--logdir T1_4xRandomT2_CombinedMixed \
--train t1_t2_paired_6875_train.csv --val t1_t2_paired_6875_val.csv \
--num_workers 2 --mask_losses --lr 1e-4 --mask_lr 1e-4 --smooth_weight 100 \
--protocals T2 T1 --mask equispaced --aux_aug None --sparsity 0.25 \
--epoch 20000 --batch_size 4 --reg Mixed \
--intel_stop 5e4 --coils 1 \
--copy_mask T1_4xRandomT2_GANOnly/ckpt/best.pth #--prefetch
```

### Evaluation
```bash
mkdir save
python3 eval.py \
--resume T1_4xRandomT2_CombinedMixed/ckpt/best.pth \
--save save
--val t1_t2_paired_6875_val.csv \
--protocals T2 T1 --aux_aug None
```
