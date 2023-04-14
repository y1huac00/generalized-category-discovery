SAVE_DIR=experiments/cifar10/
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python CCD.py \
            --dataset_name 'cifar10' \
            --batch_size 256 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 2 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2'\
            --num_unlabeled_classes_predicted 100 \
            --warmup 20 \
            --conf_new_supcon 'True' \
            --mini 'True' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out
