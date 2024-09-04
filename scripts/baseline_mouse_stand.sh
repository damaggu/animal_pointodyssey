export MUJOCO_GL=egl

python train.py \
--env mouse-stand \
--log-directory logs \
--num-timesteps 100_000_000 \
--lr 0.0001 \
--eval-freq 250_000 \
--algorithm DDPG \
--n-envs 32 \
--batch-size 64 \
--parallel \
--buffer-size 1_000_000 \
--net-arch-pi 300 200 \
--net-arch-qf 400 300 \
--tau 0.001 \
--action-noise ornstein-uhlenbeck \
--sigma 0.3 \
--theta 0.15