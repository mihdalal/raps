import os

num_seeds = 3
for framestack in [1, 2, 3]:
    for data_augs in ["translate", "no_aug", "crop"]:
        for discount in [0.99, 0.8]:
            for lr in [2e-4]:
                if data_augs == "crop":
                    pre_transform_image_size = 100
                    image_size = 84
                elif data_augs == "translate":
                    pre_transform_image_size = 100
                    image_size = 108
                else:
                    pre_transform_image_size = 84
                    image_size = 84
                os.system(
                    "python kitchen_train.py \
                    --encoder_type pixel --work_dir data/kitchen/ \
                    --env_class hinge_cabinet \
                    --action_repeat 1 --num_eval_episodes 5 \
                    --pre_transform_image_size {pre_transform_image_size} --image_size {image_size} \
                    --data_augs {data_augs} --discount {discount} --init_steps 2500 \
                    --agent rad_sac --frame_stack {framestack} --save_tb\
                    --seed -1 --critic_lr {lr} --actor_lr {lr} --encoder_lr {lr} --eval_freq 1000 --batch_size 512 --num_train_steps 200000".format(
                        data_augs=data_augs,
                        framestack=framestack,
                        discount=discount,
                        image_size=image_size,
                        lr=lr,
                        pre_transform_image_size=pre_transform_image_size,
                    )
                )
