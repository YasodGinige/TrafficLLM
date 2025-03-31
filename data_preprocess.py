from preprocessor import Data_Preprocess
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import _pickle as cPickle
import gc
import argparse
import sys
import shutil

def preprocess(args):
    data_path = args.data_path
    dataset = args.dataset
    llm_model = args.model

    if not os.path.exists("./temp_dir"):  
        os.makedirs("temp_dir")
    else:
        for filename in os.listdir("temp_dir"):
            file_path = os.path.join("temp_dir", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    Data_preprocessor = Data_Preprocess()
    Data_preprocessor.preprocess_dataset(data_path, dataset, llm_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data preprocessing")
    parser.add_argument("--data_path", type=str, default='./data', help="path to the datasets")
    parser.add_argument("--dataset", type=str, default='AWF', help="Dataset name")
    parser.add_argument("--model", type=str, default='GPT2', help="LLM name")
    args = parser.parse_args()
    preprocess(args)
