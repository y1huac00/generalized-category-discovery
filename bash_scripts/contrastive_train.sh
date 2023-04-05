PYTHON='/userhome/cs/yihuac/anaconda3/envs/gcd/bin/python'
hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=experiments/testing/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.contrastive_training.contrastive_training \
            --dataset_name 'cifar10' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2' \
            --warmup_model_dir '/userhome/cs/yihuac/finalproject/generalized-category-discovery/experiments/testing/metric_learn_gcd/log/(04.04.2023_|_45.189)/checkpoints/model.pt' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out