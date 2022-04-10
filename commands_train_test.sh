TRAIN="${PWD}/fastMRI_brain_DICOM/t1_t2_paired_6875_train.csv"
VAL="${PWD}/fastMRI_brain_DICOM/t1_t2_paired_6875_val.csv"
LOG_BASE="${PWD}/LOG"
mkdir -p ${LOG_BASE}
COILS=1
TGT=T2
REF=T1
FLAGS='--prefetch --force_gpu'
export CUDA_VISIBLE_DEVICES=0
mkdir -p ${LOG_BASE}


# Training

# NAME=8xEquispaced
# MASK=equispaced
# SPAR=0.125

# NAME=8xRandom
# MASK=standard
# SPAR=0.125

# NAME=4xRandom
# MASK=standard
# SPAR=0.25

NAME=4xEquispaced
MASK=equispaced
SPAR=0.25

# Single-Modal
python3 train.py --logdir ${LOG_BASE}/None_${NAME}${TGT}_PBSplineNone --train ${TRAIN} --val ${VAL} --num_workers 8 --lr 1e-4 --smooth_weight 1000 --gan_weight 0.1 --gan_sim_weight 1 --sim_weight 1 --protocals ${TGT} None --mask ${MASK} --aux_aug PBSpline --sparsity ${SPAR} --epoch 20000 --batch_size 4 --reg None --intel_stop 2e4 --coils ${COILS} ${FLAGS}
# Multi-Modal
python3 train.py --logdir ${LOG_BASE}/${REF}_${NAME}${TGT}_PBSplineNone --train ${TRAIN} --val ${VAL} --num_workers 8 --lr 1e-4 --smooth_weight 1000 --gan_weight 0.1 --gan_sim_weight 1 --sim_weight 1 --protocals ${TGT} ${REF} --mask ${MASK} --aux_aug PBSpline --sparsity ${SPAR} --epoch 20000 --batch_size 4 --reg None --intel_stop 2e4 --coils ${COILS} --resume ${LOG_BASE}/None_${NAME}${TGT}_PBSplineNone/ckpt/best.pt --load_nets net_mask ${FLAGS}
# Proposed
python3 train.py --logdir ${LOG_BASE}/${REF}_${NAME}${TGT}_PBSplineGANOnly --train ${TRAIN} --val ${VAL} --num_workers 8 --lr 1e-4 --smooth_weight 1000 --gan_weight 0.1 --gan_sim_weight 1 --sim_weight 1 --protocals ${TGT} ${REF} --mask ${MASK} --aux_aug PBSpline --sparsity ${SPAR} --epoch 20000 --batch_size 4 --reg GAN-Only --intel_stop 2e4 --coils ${COILS} --resume ${LOG_BASE}/None_${NAME}${TGT}_PBSplineNone/ckpt/best.pt --load_nets net_mask ${FLAGS}
# and
python3 train.py --logdir ${LOG_BASE}/${REF}_${NAME}${TGT}_PBSplineProposed --train ${TRAIN} --val ${VAL} --num_workers 8 --lr 1e-4 --smooth_weight 1000 --gan_weight 0.1 --gan_sim_weight 1 --sim_weight 1 --protocals ${TGT} ${REF} --mask ${MASK} --aux_aug PBSpline --sparsity ${SPAR} --epoch 20000 --batch_size 4 --reg Mixed --intel_stop 2e4 --coils ${COILS} --resume ${LOG_BASE}/${REF}_${NAME}${TGT}_PBSplineGANOnly/ckpt/best.pt --load_nets net_{mask,D,G,T} ${FLAGS}


# Testing

EVAL_BASE='${PWD}/eval'
DATA_TEST='${PWD}/fastMRI_brain_DICOM/t1_t2_paired_6875_test.csv'
AUX_AUG='-1'

function run_test(){
  echo ${NAME}
  mkdir -p ${EVAL_BASE}/${NAME}
  if test -f ${EVAL_BASE}/${NAME}/md5sum && md5sum -c ${EVAL_BASE}/${NAME}/md5sum
  then
    echo SKIPPED
  else
    python3 eval.py \
      --resume ${LOG_BASE}/${NAME}/ckpt/best.pt \
      --val ${DATA_TEST} \
      --protocals ${PROTOCALS} --aux_aug ${AUX_AUG} \
      --save ${EVAL_BASE}/${NAME} \
      --metric ${EVAL_BASE}/${NAME}.json
    md5sum ${LOG_BASE}/${NAME}/ckpt/best.pt/* > ${EVAL_BASE}/${NAME}/md5sum
  fi
}

# Single-Modal
PROTOCALS="${TGT} None"
ENAME="None_${NAME}${TGT}_PBSplineNone" run_test
# Multi-Modal
PROTOCALS="${TGT} ${REF}"
ENAME="${REF}_${NAME}${TGT}_PBSplineNone" run_test
# Proposed
ENAME="${REF}_${NAME}${TGT}_PBSplineProposed" run_test

