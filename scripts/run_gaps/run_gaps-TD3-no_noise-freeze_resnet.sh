export MUJOCO_GL=egl

python train.py \
--env rodent-run-gaps \
--log-directory logs \
--num-timesteps 100_000_000 \
--lr 0.0001 \
--eval-freq 250_000 \
--save-freq 500_000 \
--algorithm TD3 \
--n-envs 32 \
--batch-size 128 \
--buffer-size 1_000_000 \
--net-arch-pi 1024 1024 \
--net-arch-qf 1024 1024 \
--tau 0.001 \
--action-noise none \
--camera-id 0 \
--obs-wrapper none \
--feature-extractor combined \
--freeze-resnet \
--save-directory run_gaps-TD3-no_noise-freeze_resnet