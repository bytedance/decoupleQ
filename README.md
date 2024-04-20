# decoupleQ: Towards 2-bit Post-Training Uniform Quantization via decoupling Parameters into Integer and Floating Points

This repository contains the code for decoupleQ, the paper is xxx

Some of the code in this repo is built on top of [OPTQ's repository](https://github.com/IST-DASLab/gptq). We sincerely thank OPTQ for their great contribution.
Please feel free to raise issues or contact guoyi.0@bytedance.com if you have any question.

## Dependencies
All of our experiments are conducted in the following environment.
* datasets==1.17.0
* transformers==4.35.0
* torch==2.1.0


## Reproduce
```
bash run_llama.sh # will get result 9.49 for wikiText2
bash run_resnet.sh # will get result 64.134 for ResNet-18
````
## Results
Here is a summary of LLama results:


![decoupleQ](imgs/img.png)


## Cite

If you found this work useful, please consider citing: