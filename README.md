features(working):
deepspeed/ops/csrc/uio
compiler/triton

dev:
pip3 install -r requirements/requirements-dev.txt
pip3 install -r requirements/requirements-cpu.txt




https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

https://github.com/intel/intel-extension-for-pytorch
python3 -m pip install intel_extension_for_pytorch -f https://developer.intel.com/ipex-whl-stable-cpu

https://github.com/intel/torch-ccl/tree/ccl_torch2.1.0%2Bcpu
python3 -m pip install oneccl_bind_pt==2.1.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

https://github.com/intel/intel-extension-for-deepspeed
pip3 install intel-extension-for-deepspeed

PYTHONPATH

export LD_LIBRARY_PATH=/lib:/usr/lib:/home/zhf/.local/lib/python3.10/site-packages/torch/lib:/home/zhf/.local/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/lib:/home/zhf/.local/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:LD_LIBRARY_PATH
export DS_ACCELERATOR=cpu

https://intel.github.io/intel-extension-for-pytorch/cpu/2.1.0+cpu/
https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installations/linux.html
https://intel.github.io/intel-extension-for-pytorch/latest/tutorials/performance_tuning/tuning_guide.html

ref:

训练[kubeflow, flink, deepspeed, 存储]
[kubeflow](https://www.kubeflow.org/)
[Volcano](https://volcano.sh/) 
[flink](https://github.com/flink-extended/dl-on-flink)
[deepspeed](git clone https://github.com/microsoft/DeepSpeed.git)
[deepspeed-chat](https://github.com/microsoft/DeepSpeedExamples.git)

apps
[search](https://hub.baai.ac.cn/view/23202)
[llama2]()
平台
[nvidia ai enterprise](https://docs.nvidia.com/ai-enterprise/1.1/user-guide/index.html)

压缩[蒸馏, 剪枝, 稀疏, 量化]

算法[NLP, CV, 语音, 推荐系统](GPT, Transformer, diffusion, NeRF, GAN, MoE, Conformer, Tacontron2, DLRM):

编译器[riscv, arm, xeon, x86, p40][llvm, tvm, glow, ggml, cuda, vllm]
[LLVM IR](https://llvm.org/docs/LangRef.html)
[TVM](https://daobook.github.io/tvm/)
[TVM IR](https://tvm.apache.org/docs/reference/api/python/ir.html)
[openvino](https://docs.openvino.ai/2023.0/documentation.html)
[glow](https://github.com/openai/glow)
[vllm](https://vllm.ai/?continueFlag=24b2e01413fd53e24a2779b4a664ca16)

芯片[riscv+ai]

misc[小样本学习/迁移学习, 复杂问题推理, 多模态, 微调]

openai[prompt, RLHF]
[key papers](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
[openai api](https://juejin.cn/post/7225126264663605309)







