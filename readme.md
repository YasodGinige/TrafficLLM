## TrafficLLM: LLMs for Improved Open-Set Encrypted Traffic Analysis

![](https://img.shields.io/badge/license-MIT-000000.svg)
[![arXiv](https://img.shields.io/badge/arXiv-1909.05658-<color>.svg)]()

**Note:**
- ⭐ **Please leave a <font color='orange'>STAR</font> if you like this project!** ⭐
- If you are using this work for academic purposes, please cite our [paper](https://dl.acm.org/doi/abs/10.1145/3674213.3674217).
- If you find any <font color='red'>incorrect</font> / <font color='red'>inappropriate</font> / <font color='red'>outdated</font> content, please kindly consider opening an issue or a PR.

<div align="center">
    <img src="/Images/models.png" width="850" height="450" alt="overall architecure"/>
</div>

In this repository, we guide you in setting up the TrafficLLM project in a local environment and reproducing the results. TrafficLLM, a novel traffic analysis attack that leverages GPT-2, a popular LLM, to enhance feature extraction, thereby improving
the open-set performance of downstream classification. We use five existing encrypted traffic datasets to show how the feature extraction by GPT-2 improves the open-set performance of traffic
analysis attacks. As the open-set classification methods, we use K-LND, OpenMax, and Backgroundclass methods, and shows that K-LND methods have higher performance overall.

**Datasets:** [AWF](https://arxiv.org/abs/1708.06376), [DF](https://arxiv.org/abs/1801.02265), [DC](https://www.semanticscholar.org/paper/Deep-Content%3A-Unveiling-Video-Streaming-Content-Li-Huang/f9feff95bc1d68674d5db426053f417bd2c8786b), [USTC](https://drive.google.com/file/d/1F09zxln9iFg2HWoqc6m4LKFhYK7cDQv_/view), [CSTNet-tls](https://drive.google.com/drive/folders/1JSsYmevkxQFanoKOi_i1ooA6pH3s9sDr)

**Openset methods**
- [K-LND methods](https://github.com/ThiliniDahanayaka/Open-Set-Traffic-Classification)
- OpenMax
- Background class

# Using TrafficLLM

First, clone the git repo and install the requirements.
```
git clone https://github.com/YasodGinige/TrafficLLM.git
cd TrafficLLM
pip install -r requirements.txt
```
Next, download the dataset and place it in the data directory.
```
gdown https://drive.google.com/uc?id=1-MVfxyHdQeUguBmYrIIw1jhMVSqxXQgO
unzip data.zip 
```

Then, preprocess the dataset you want to train and evaluate. Here, the dataset name should be DF, AWF, DC, USTC, or CSTNet, and the model_name should be either GPT2 or LLaMA.
```
python3 data_preprocess.py --data_path ./data --dataset <dataset_name> --model <model_name>
```

### GPT-2 Fine-tuning

To fine-tune the model, run the suitable code for the dataset:
```
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 60  --dataset DF
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 200  --dataset AWF
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 4  --dataset DC
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 12  --dataset USTC
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 75  --dataset CSTNet
```
To evaluate, run the suitable code for the dataset:
```
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 60 --K_number 30 --TH_value 0.8 --dataset DF
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 200 --K_number 50 --TH_value 0.9 --dataset AWF
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 4 --K_number 4 --TH_value 0.9 --dataset DC
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 12 --K_number 5 --TH_value 0.8 --dataset USTC
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 5 --num_labels 75 --K_number 20 --TH_value 0.8 --dataset CSTNe
```

You can find the fine-tuned models [here](https://drive.google.com/drive/folders/1aln2WG_XrRzPZUym44uPew7-b8stL9T-?usp=sharing).

### LLaMA Fine-tuning

To fine-tune the LLaMA model and obtain results, run the following commands accordingly.
```
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 3 --num_labels 60  --dataset DF --K_number 30 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 3 --num_labels 200  --dataset AWF --K_number 50 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 8 --epochs 2 --num_labels 4  --dataset DC --K_number 4 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 8 --epochs 2 --num_labels 12  --dataset USTC --K_number 5 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 2 --num_labels 75  --dataset CSTNet --K_number 20 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 2 --num_labels 60  --dataset IoT --K_number 30 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 8 --epochs 2 --num_labels 10  --dataset ISCX --K_number 5 --TH_value 0.9
```

## Attention Maps
Attention plots of GPT-2 and ET-BERT models for AWF and IoT traffic traces are given below. Note that the GPT-2 model pays attention to critical patterns in the traffic trace (even for open-set-unseen data), while ET-BERT's attention is widespread. This suggests that GPT-2 can learn generalized features following the traffic trace and paying attention to correct points.
<div align="center">
    <img src="/Images/attention_maps.png" width="800" height="850" alt="overall architecure"/>
</div>

# Citations
If you are using this work for academic purposes, please cite our [paper](https://dl.acm.org/doi/abs/10.1145/3674213.3674217).
```
citation here
```
