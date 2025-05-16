# VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service


## Brief Introduction

we present the first systematic evaluation of the efficiency robustness of VLMs under the black-box setting. We introduce VLMInferSlow, a novel black-box attack framework designed to reduce the inference efficiency of VLMs.


## Installation

This code is tested on our local environment (python=3.9.20, cuda=12.1, torch=2.1.0)


Install required packages:

```bash
pip install -r requirements.txt
```

This code uses Flamingo model. Install the related packages:

```bash
git clone https://github.com/dhansmair/flamingo-mini.git
cd flamingo-mini
pip install .
```

We use the model from huggingface: https://huggingface.co/dhansmair/flamingo-tiny


## VLMInferSlow

Run the following command to generate VLMInferSlow adversarial image

```shell
bash run.sh
```
