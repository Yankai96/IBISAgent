set -x
model=/mnt/shared-storage-user/sdpdev-fs/sunhaoran/train/segagent/sft/qwen2_5vl-7b-lora/merge4415
project_name='IBIS'
experiment_name='Qwen2_5-7B-grpo-512-v3' # 不同实验记得换，方便tb监控
reward=/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/med_seg.py
CKPTS_DIR=/mnt/shared-storage-user/sdpdev-fs/sunhaoran/train/segagent/grpo/qwen2_5vl-7b/${experiment_name}


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=['/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_0.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_1.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_2.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_4.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_5.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_6.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_8.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_9.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_10.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_12.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_13.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_14.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_16.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_17.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_18.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_20.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_21.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_22.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_24.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_train_25.parquet']" \
    "data.val_files=['/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/organ/merged_test_0.parquet','/mnt/shared-storage-user/sdpdev-fs/sunhaoran/segagent/train/data/biomed512/merged_test_0.parquet']" \
    data.train_batch_size=256 \
    data.max_prompt_length=30000 \
    data.max_response_length=2768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$reward \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=8 \
    trainer.total_epochs=12 \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.log_val_generations=2 \
    +trainer.remove_previous_ckpt_in_save=False \
    $@
