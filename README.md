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

训练[kubeflow, flink, deepspeed, 存储]
[kubeflow](https://www.kubeflow.org/)
[Volcano](https://volcano.sh/) 
[flink](https://github.com/flink-extended/dl-on-flink)

apps
[search](https://hub.baai.ac.cn/view/23202)

平台
[nvidia ai enterprise](https://docs.nvidia.com/ai-enterprise/1.1/user-guide/index.html)

压缩[蒸馏, 剪枝, 稀疏, 量化]

算法[NLP, 多模态, CV, 语音, 推荐系统](GPT, Transformer, diffusion, NeRF, GAN, MoE, Conformer, Tacontron2, DLRM):

编译器[riscv, arm, xeon, x86, p40][llvm, tvm, glow, ggml, cuda, vllm]
[LLVM IR](https://llvm.org/docs/LangRef.html)
[TVM](https://daobook.github.io/tvm/)
[TVM IR](https://tvm.apache.org/docs/reference/api/python/ir.html)
[openvino](https://docs.openvino.ai/2023.0/documentation.html)
[glow](https://github.com/openai/glow)
[vllm](https://vllm.ai/?continueFlag=24b2e01413fd53e24a2779b4a664ca16)

芯片[riscv+ai]

misc[小样本训练, 微调]

openai[prompt, RLHF]
[key papers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
[openai api](https://juejin.cn/post/7225126264663605309)
[State of GPT: A Programmer's Perspective](https://mp.weixin.qq.com/s?__biz=MzI4MjkzMDIwMg==&mid=2247483746&idx=1&sn=c5c577aad0bcfd0e339669d30e9a4fdc&scene=21#wechat_redirect)






