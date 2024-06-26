# decoupleQ: Towards 2-bit Post-Training Uniform Quantization via decoupling Parameters into Integer and Floating Points

This repository contains the code for decoupleQ, the paper link is https://arxiv.org/abs/2404.12759 

The W2 CUDA kernel is available at https://github.com/NVIDIA/TensorRT-LLM/pull/1568

Some of the code in this repo is built on top of [OPTQ's repository](https://github.com/IST-DASLab/gptq). We sincerely thank OPTQ for their great contribution.

Please feel free to raise issues or contact chenwei.gavin@bytedance.com or guoyi.0@bytedance.com if you have any question.

## Dependencies
All of our experiments are conducted in the following environment.
* datasets==1.17.0
* transformers==4.35.0
* torch==2.1.0


## Reproduce
To reproduce the results of LLama, you should first download the models from [here](https://llama.meta.com/llama-downloads/), 
then put it at ``MODEL_PATH``. Change the ``MODEL_PATH`` in the following command to the destination where the models are placed.
```
bash run_llama.sh MODEL_PATH # will get result 9.49 for wikiText2
bash run_resnet.sh # will get result 64.134 for ResNet-18
````
In llama quantization, if you find that the reproduced results (including the runtime) are far from the reported results, 
consider modifying the flag: `torch.backends.cuda.matmul.allow_tf32`. More details can be found in [here](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).

to run inference demo, you should first modify the ``build.sh``, change the ``DCMAKE_PREFIX_PATH``, ``DDECOUPLEQ_TORCH_HOME``, 
``DDECOUPLEQ_CUDA_HOME`` and ``DDECOUPLEQ_CUDNN_HOME`` based on your system, and then run the following commands:
```
git submodule update --init
bash build.sh  # need cmake3.21+
bash run_inference_llama.sh $LLAMA_ORG_MODEL_DIR $LLAMA_TRUE_QUANT_MODEL_PT
```


## Results
Here is a summary of LLama results (runtime for
the quantization process is measured in hours):

![decoupleQ](imgs/img.png)


## Updates
Here is the results of ByteDance's two ASR models. The models are quantized into W2A16g64.
In decoupleQ+sft, when the whole model is quantized, we fine-tune the float-point parts with labeled dataset, while freezing all the
integer part. There are two sub-domains in task B, and we report the WER of both. (runtime is measured in hours)

![decoupleQ](imgs/private_exp.png)

## Cite

If you found this work useful, please consider citing: 
```
@article{guo2024decoupleq,
  title={decoupleQ: Towards 2-bit Post-Training Uniform Quantization via decoupling Parameters into Integer and Floating Points},
  author={Guo, Yi and Kong, Fanliu and Li, Xiaoyang and Li, Hui and Chen, Wei and Tian, Xiaogang and Cai, Jinping and Zhang, Yang and Liu, Shouda},
  journal={arXiv preprint arXiv:2404.12759},
  year={2024}
}
```
