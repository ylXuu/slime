#!/bin/bash

# Training script for ASearcher on slime with Qwen3-4B-Instruct-2507
# Local retrieval (Wikipedia) + visit (page access)

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# pkill -9 python
sleep 3
pkill -9 ray
# pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


if command -v nvidia-smi >/dev/null 2>&1; then
   DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
   DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
   NUM_GPUS=8
fi
echo "NUM_GPUS: $NUM_GPUS"


# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"
source "/cpfs01/yilong.xu/slime/scripts/models/qwen3-4B-Instruct-2507.sh"


TORCH_DIST_PATH="/cpfs01/models/Qwen/Qwen3-4B-Instruct-2507_torch_dist"
if [ -d "$TORCH_DIST_PATH" ]; then
   echo "Detected existing torch_dist directory at $TORCH_DIST_PATH."
else
   echo "No existing torch_dist directory found at $TORCH_DIST_PATH. Starting conversion from Hugging Face format."
   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint /cpfs01/models/Qwen/Qwen3-4B-Instruct-2507 \
      --save "$TORCH_DIST_PATH"
fi



now=$(date "+%Y%m%d-%H%M%S")
exp_name="qwen3-4b-instruct-2507-asearcher-${now}"

CKPT_ARGS=(
   --hf-checkpoint /cpfs01/models/Qwen/Qwen3-4B-Instruct-2507 # 只用于找tokenizer, 实际不会读这个模型
   --ref-load ${TORCH_DIST_PATH}
   --load /cpfs01/yilong.xu/ckpt/$exp_name
   --save /cpfs01/yilong.xu/ckpt/$exp_name
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data /cpfs01/yilong.xu/datasets/ASearcher-slime/train/ASearcher-Base-35k.jsonl
   --input-key prompt
   --label-key label
   # TODO: Do NOT use --apply-chat-template; we construct the chat format inside generate().
   --rollout-shuffle
   --log-multi-turn
   --num-rollout 1000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048 # 单次请求
   --rollout-temperature 1

   --global-batch-size 256 # 32 * 8
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   # --eval-prompt-data 2wiki /cpfs01/yilong.xu/datasets/ASearcher-slime/test/2WikiMultihopQA_rand1000.jsonl musique /cpfs01/yilong.xu/datasets/ASearcher-slime/test/Musique_rand1000.jsonl frames /cpfs01/yilong.xu/datasets/ASearcher-slime/test/frames.jsonl bamboogle /cpfs01/yilong.xu/datasets/ASearcher-slime/test/Bamboogle.jsonl nq /cpfs01/yilong.xu/datasets/ASearcher-slime/test/NQ_rand1000.jsonl popqa /cpfs01/yilong.xu/datasets/ASearcher-slime/test/PopQA_rand1000.jsonl hotpotqa /cpfs01/yilong.xu/datasets/ASearcher-slime/test/HotpotQA_rand1000.jsonl triviaqa /cpfs01/yilong.xu/datasets/ASearcher-slime/test/TriviaQA_rand1000.jsonl
   # --eval-prompt-data 2wiki /cpfs01/yilong.xu/datasets/ASearcher-slime/test/2WikiMultihopQA_rand1000.jsonl musique /cpfs01/yilong.xu/datasets/ASearcher-slime/test/Musique_rand1000.jsonl nq /cpfs01/yilong.xu/datasets/ASearcher-slime/test/NQ_rand1000.jsonl hotpotqa /cpfs01/yilong.xu/datasets/ASearcher-slime/test/HotpotQA_rand1000.jsonl
   --eval-prompt-data hotpotqa /cpfs01/yilong.xu/datasets/ASearcher-slime/test/HotpotQA_rand1000.jsonl
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt 1
   # --eval-max-response-len 131072
   --eval-top-k 1
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216 # 这个值要大于max-response-len。因为如果max-response-len更大，这个数据在训练时并不会被删去，而是会自己组成一个batch放到一张卡上，然后可能会导致OOM
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   # --lr-warmup-iters 100     # 建议增加一段预热，让训练更稳
   --min-lr 1e-7
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group asearcher_qwen3-4B
   # --wandb-key ${WANDB_KEY}
   --use-tensorboard
   --tb-project-name ./tensorboard_log
   --tb-experiment-name $exp_name
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5 # 训推一体化的时候，megatron初始化会占一些显存，初始化完成后才会释放（offload）。因此需要调整 --sglang-mem-fraction-static 参数，降低 SGLang 的 GPU 显存占用比例，以避免 GPU 显存不足的问题。
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_asearcher.generate
   --custom-rm-path reward_func.reward_func
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --num-cpus 64 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_ALGO\": \"Ring\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
    \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
