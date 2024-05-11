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
then put it at ``PATH``. Change the ``PATH`` in the run_llama.sh to the destination where the models are placed.
```
bash run_llama.sh # will get result 9.49 for wikiText2
bash run_resnet.sh # will get result 64.134 for ResNet-18
````
In llama quantization, if you find that the reproduced results (including the runtime) are far from the reported results, 
consider modifying the flag: `torch.backends.cuda.matmul.allow_tf32`. More details can be found in [here](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere).


## Results
Here is a summary of LLama results:


![decoupleQ](imgs/img.png)


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