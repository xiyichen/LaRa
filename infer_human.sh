CUDA_VISIBLE_DEVICES=0 python evaluation.py configs/infer.yaml \
       infer.ckpt_path=ckpts/epoch=29.ckpt \
       infer.save_folder=outputs/dna-rendering/ \
       infer.dataset.dataset_name=human \
    #    infer.save_mesh=False \