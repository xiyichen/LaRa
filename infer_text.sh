python evaluation.py configs/infer.yaml \
       infer.ckpt_path=ckpts/epoch=29.ckpt \
       infer.save_folder=outputs/single-view/ \
       infer.dataset.generator_type="zero123plus-v1.2" \
       infer.dataset.image_pathes=\["assets/examples/human.png"\]