dataset="mskimpact_noclinical"

CUDA_VISIBLE_DEVICES=0 python main.py  \
   --max_epochs=10  --num_workers=8 \
   --model_name_or_path  '/userfs/lambdaKG/models/t5-base' \
   --model_class T5KGC \
   --strategy="deepspeed_stage_2" \
   --lit_model_class KGT5LitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --precision 16 \
   --batch_size 64 \
   --accumulate_grad_batches 1 \
   --prefix_tree_decode 1 \
   --check_val_every_n_epoch 1 \
   --wandb \
   --use_ce_loss 1 \
   --dataset ${dataset} \
   --eval_batch_size 4 \
   --beam_size 10 \
   --max_seq_length 128 \
   --lr 1e-4 \
   --wandb_name 'mskimpact_noclinical_t5_noinv'


CUDA_VISIBLE_DEVICES=0 python main.py  \
   --max_epochs=10  --num_workers=8 \
   --model_name_or_path  '/userfs/lambdaKG/models/bart-base' \
   --model_class BartKGC \
   --strategy="deepspeed_stage_2" \
   --lit_model_class KGBartLitModel \
   --label_smoothing 0.1 \
   --data_class KGT5DataModule \
   --precision 16 \
   --batch_size 64 \
   --accumulate_grad_batches 1 \
   --prefix_tree_decode 1 \
   --check_val_every_n_epoch 1 \
   --wandb \
   --use_ce_loss 1 \
   --dataset ${dataset} \
   --eval_batch_size 4 \
   --beam_size 10 \
   --max_seq_length 128 \
   --lr 1e-4 \
   --wandb_name 'mskimpact_noclinical_bart_noinv'


#note I removed the prompt option, cuz threw some errors
#also modified bartlocation
#change model path, class, litmodel class, and potentially data class