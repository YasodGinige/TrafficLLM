import torch
import pandas as pd
import numpy as np
from peft import LoraConfig, PeftModel
from transformers import AutoModelForSequenceClassification,AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from trl import SFTTrainer
import torch.nn as nn
import sys
import gc
import time
import random
import argparse

random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "./trained_LlaMA_models/DC"
cache_dir='/mnt/sdrive/llama_guest'             #change this directory location to your local LLaMA location
hf_token = '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'

lora_alpha = 32
lora_dropout = 0.05
lora_r = 8


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return (len(self.texts)-1)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['text'],
        max_length=max_len,
        padding='max_length',
        truncation=True
    )
    # Add labels properly for loss computation
    model_inputs['labels'] = examples['target']
    return model_inputs

def pre_process(dataset):
    dataset['trace'] = dataset['text']
    return dataset

def get_labels(file):
    df = pd.read_csv(file)
    return np.array(df['target'])

# Function for training
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        if loss.dim() > 0:
            loss = loss.mean() 

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

# Function for validation 
def validate(model, dataloader, device):
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            if loss.dim() > 0:
                loss = loss.mean() 

            total_loss += loss.item()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

def predict(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    
    for ind,batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            if ind == 0:
                predictions_labels = logits.to('cpu').numpy()
            else:
                predictions_labels = np.concatenate((predictions_labels, logits.to('cpu').numpy()), axis=0)
    return predictions_labels

def create_model_and_tokenizer(num_labels):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=num_labels,  output_hidden_states=True,use_auth_token=hf_token, cache_dir=cache_dir, use_safetensors = True,
        quantization_config = bnb_config,
        trust_remote_code = True,
        device_map = "auto")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def calculate_thresholds(NB_CLASSES, model_predictions, y_valid, Mean_Vectors, K_number, TH_value):
    #model_predictions = args.model.predict(X_valid)

    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    Indexes=[]
    for i in range(NB_CLASSES):
        Indexes.append([])

    Values={}
    for i in range(NB_CLASSES):
        Values[i]=[0]*NB_CLASSES


    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]):
                    Values[y_valid[i]][k]+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist

    for i in range(NB_CLASSES):
        Tot=0
        for l in range(K_number):
            Min=min(Values[i])
            Tot+=Min
            Indexes[i].append(Values[i].index(Min))
            Values[i][Values[i].index(Min)]=1000000

    Indexes=np.array(Indexes)

    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*TH_value)]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_1=np.array(TH)
    print("Thresholds for method 1 calculated")
    
    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot+=(np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist)
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*(1-TH_value))]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_2=np.array(TH)
    print("Thresholds for method 2 calculated")
    
    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])
            Tot=dist/Tot
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*TH_value)]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_3=np.array(TH)
    print("Thresholds for method 3 calculated")
    
    return Threasholds_1, Threasholds_2, Threasholds_3, Indexes


def final_classification(NB_CLASSES, model_predictions_test, model_predictions_open, y_test, y_open, Mean_vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3):
     
    
    print("\n", "############## Distance Method 1 #################################")
    prediction_classes=[]
    for i in range(len(model_predictions_test)):

        d=np.argmax(model_predictions_test[i], axis=0)
        if np.linalg.norm(model_predictions_test[i]-Mean_vectors[d])>Threasholds_1[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)
    prediction_classes=np.array(prediction_classes)

    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):

        d=np.argmax(model_predictions_open[i], axis=0)
        if np.linalg.norm(model_predictions_open[i]-Mean_vectors[d])>Threasholds_1[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
    prediction_classes_open=np.array(prediction_classes_open)
    
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES)

    print("\n", "############## Distance Method 2 #################################")
    prediction_classes=[]
    for i in range(len(model_predictions_test)):
        d=np.argmax(model_predictions_test[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_test[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_test[i])-dist

        if Tot<Threasholds_2[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)

    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):
        d=np.argmax(model_predictions_open[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d]-model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_open[i])-dist

        if Tot<Threasholds_2[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)

    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES)

    
    print("\n", "############## Distance Method 3 #################################")

    prediction_classes=[]
    for i in range(len(model_predictions_test)):
        d=np.argmax(model_predictions_test[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_test[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_test[i])

        Tot=dist/Tot
        if Tot>Threasholds_3[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)

    prediction_classes=np.array(prediction_classes)
    
    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):
        d=np.argmax(model_predictions_open[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_open[i])
        Tot=dist/Tot
        if Tot>Threasholds_3[d]:
            prediction_classes_open.append(NB_CLASSES)

        else:
            prediction_classes_open.append(d)

    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES)


def main(args):
    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs
    num_labels = args.num_labels
    dataset = args.dataset
    K_number = args.K_number
    TH_value = args.TH_value

    if not os.path.exists("./trained_models"):  
        os.makedirs("trained_models") 

    train_dataset = pd.read_csv('./temp_dir/train.csv')
    val_dataset = pd.read_csv('./temp_dir/valid.csv')

    print('Loading LLaMA model')
    peft_config = LoraConfig(
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        r = lora_r,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    model, tokenizer = create_model_and_tokenizer(num_labels)

    model.add_adapter(peft_config)

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,          
        num_train_epochs=epochs,             
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=8,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",           
        logging_steps=10,
        learning_rate=1e-4
    )

    train_dataset = pd.read_csv('./temp_dir/train.csv')
    val_dataset = pd.read_csv('./temp_dir/valid.csv')

    train_texts = train_dataset['text']
    train_labels = train_dataset['target']

    validation_texts = val_dataset['text']
    validation_labels = val_dataset['target']

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    eval_dataset = TextDataset(validation_texts, validation_labels, tokenizer, max_len=max_len)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        data_collator=data_collator,
    )

    trainer.train()

    test_dataset = pd.read_csv('./temp_dir/test.csv')
    open_dataset = pd.read_csv('./temp_dir/open.csv')

    train_dataset = Dataset.from_pandas(dataset)
    eval_dataset = Dataset.from_pandas(val_dataset)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create a DataLoader for the test set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
    valid_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
    open_loader = DataLoader(open_dataset, batch_size=batch_size, collate_fn=data_collator, drop_last=True)

    # Put the model in evaluation mode
    model.eval()

    latent_vectors_train = []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass, get raw output from the last hidden layer
            start = time.time()
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            end = time.time()
            print(len(batch))
            print("per input inference time:", (end-start)/len(batch))
            sys.exit()
            
            logits = outputs.logits  
            latent_vectors_train.append(logits.cpu().numpy())

    latent_vectorst_train = np.array(latent_vectors_train)
    latent_vectorst_train = latent_vectorst_train.reshape(latent_vectorst_train.shape[0]*latent_vectorst_train.shape[1], -1)
    train_labels = np.array(train_dataset.labels).astype(int)
    Mean_Vectors = claculate_mean_vectors(num_labels, latent_vectorst_train, train_labels)

    del latent_vectorst_train, train_labels
    gc.collect()

    latent_vectors_valid = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass, get raw output from the last hidden layer
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits  
            latent_vectors_valid.append(logits.cpu().numpy())

    latent_vectors_valid = np.array(latent_vectors_valid)
    latent_vectors_valid = latent_vectors_valid.reshape(latent_vectors_valid.shape[0]*latent_vectors_valid.shape[1], -1)
    valid_labels = np.array(eval_dataset.labels).astype(int)
    Threasholds_1, Threasholds_2, Threasholds_3, Indexes = calculate_thresholds(num_labels, latent_vectors_valid, valid_labels, Mean_Vectors, K_number, TH_value)

    del latent_vectors_valid, valid_labels
    gc.collect()

    latent_vectors = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass, get raw output from the last hidden layer
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits  
            latent_vectors.append(logits.cpu().numpy())

    latent_vectors_open = []

    with torch.no_grad():
        for batch in open_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass, get raw output from the last hidden layer
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits 
            latent_vectors_open.append(logits.cpu().numpy())

    latent_vectors_test = np.array(latent_vectors)
    latent_vectors_open = np.array(latent_vectors_open)

    latent_vectors_test = latent_vectors_test.reshape(latent_vectors_test.shape[0]*latent_vectors_test.shape[1], -1)
    latent_vectors_open = latent_vectors_open.reshape(latent_vectors_open.shape[0]* latent_vectors_open.shape[1], -1)

    test_labels = np.array(test_dataset.labels).astype(int)
    open_labels = np.array([num_labels]*len(open_dataset.labels))
    final_classification(num_labels, latent_vectors_test, latent_vectors_open, test_labels, open_labels, Mean_Vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model with sequence classification")
    parser.add_argument("--max_len", type=int, default=1024, help="Max length of the text for input")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--num_labels", type=int, default=120, help="Number of labels for classification")
    parser.add_argument("--dataset", type=str, default='AWF', help="Dataset name")
    parser.add_argument("--K_number", type=int, default=30, help="k value in k-LND")
    parser.add_argument("--TH_value", type=float, default=0.9, help="Threshold")

    args = parser.parse_args()
    main(args)

