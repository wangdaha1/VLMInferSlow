# VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service

This repository provides the official code of our ACL 2025 work: [VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service] 

Link: https://arxiv.org/abs/2506.15755


## Brief Introduction

we present the first systematic evaluation of the efficiency robustness of VLMs under the black-box setting. We introduce VLMInferSlow, a novel black-box attack framework designed to reduce the inference efficiency of VLMs.


## Installation

This code is tested on our local environment (python=3.9.20, cuda=12.1, torch=2.1.0)


Install required packages:

```bash
pip install -r requirements.txt
```

This code uses the Flamingo model. Install the related packages:

```bash
git clone https://github.com/dhansmair/flamingo-mini.git
cd flamingo-mini
pip install .
```

We use the model from Huggingface: https://huggingface.co/dhansmair/flamingo-tiny


## Run the code

Run the following command to generate the VLMInferSlow adversarial image

```shell
bash run.sh
```


## Citation

If you find this work helpful, please cite as follows.

```
@article{wang2025vlminferslow,
  title={VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service},
  author={Wang, Xiasi and Yao, Tianliang and Chen, Simin and Wang, Runqi and Ye, Lei and Gao, Kuofeng and Huang, Yi and Yao, Yuan},
  journal={Association for Computational Linguistics},
  year={2025}
}
```

