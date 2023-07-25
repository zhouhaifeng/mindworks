doc:
featurelist


feature:
deepspeed/ops/csrc/io_uring
deepspeed/ops/csrc/rdma
compiler/llvm

训练流程:
1: 调用api, 启动工作流任务进行训练
2: 在计算节点使用flink进行训练
# Stream Training
./bin/flink run \
  -py "${DL_ON_FLINK_DIR}"/examples/linear/pytorch/flink_train.py \
  --jarfile "${DL_ON_FLINK_DIR}"/lib/dl-on-flink-pytorch-0.5.0-jar-with-dependencies.jar \
  --model-path "${MODEL_PATH}"
  
# Batch Training with 1280 samples for 100 epochs
./bin/flink run \
  -py "${DL_ON_FLINK_DIR}"/examples/linear/pytorch/flink_train.py \
  --jarfile "${DL_ON_FLINK_DIR}"/lib/dl-on-flink-pytorch-0.5.0-jar-with-dependencies.jar \
  --model-path "${MODEL_PATH}" \
  --epoch 100 \
  --sample-count 1280


ref:
[openai api](https://juejin.cn/post/7225126264663605309)