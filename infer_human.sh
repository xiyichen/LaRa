CUDA_VISIBLE_DEVICES=0 python evaluation.py configs/infer.yaml \
       infer.save_folder=outputs/dna-rendering_finetune_0.3_new_eval_all/ \
       infer.dataset.dataset_name=human \
       infer.video_frames=0 \
       infer.save_mesh=False \
       infer.metric_path=outputs/metrics/dna_rendering_finetune_0.3_new_eval_all \
       infer.ckpt_path=/fs/gamma-projects/3dnvs_gamma/LaRa/logs/LaRa/finetune_0.3_new_eval_all/epoch=9.ckpt \
       # infer.ckpt_path=./ckpts/epoch=29.ckpt \