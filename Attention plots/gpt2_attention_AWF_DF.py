import itertools
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from matplotlib.gridspec import GridSpec
from spacy.symbols import nsubj, VERB
from tqdm.auto  import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    set_seed,
    TrainingArguments,
    Trainer,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    GPT2ForSequenceClassification
)

if torch.cuda.is_available():
    device = torch.cuda.current_device()
torch.cuda.empty_cache()

def visualize_single(att_map, tokens, n_layer, n_head):
    """
    Attention map for a given layer and head
    """
    plt.figure(figsize=(16, 12))
    crop_len = len(tokens)
    plt.imshow(att_map[n_layer, n_head, :crop_len, :crop_len], cmap='Reds')
    plt.xticks(range(crop_len), tokens, rotation=60, fontsize=12)
    plt.yticks(range(crop_len), tokens, fontsize=12)

    plt.grid(False)
def visualize_all(attn, crop_len, n_layers=12, n_heads=12, title=""):
    """
    Full grid of attention maps [12x12]
    """
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(15, 12), sharex=True, sharey=True)

    for i in range(n_layers):
        for j in range(n_heads):
            im = axes[i, j].imshow(attn[i, j, :crop_len, :crop_len], cmap='Oranges')
            axes[i, j].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle(title, fontsize=20)
def visualize_before_and_after(before, after, title='', cmap="Greens"):
    """
    Visualize the difference between base BERT and fine-tuned BERT
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax1, ax2 = axes[0], axes[1]

    vmax = max(np.max(before), np.max(after))

    im = ax1.imshow(before, cmap=cmap, vmax=vmax)
    ax1.set_title('Base model')
    ax1.grid(False)

    im = ax2.imshow(after, cmap=cmap, vmax=vmax)
    ax2.set_title('Fine-tuned model')
    ax2.grid(False)

    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle(title, fontsize=20)
# See spacy docs for tag-pos relation

def get_max_target_weight(attn, target_indices):
    """
    Get the maximum attn weight out of target tokens (given by their indices)
    """
    if not target_indices:
        return 0
    avg_attn = np.mean(attn, axis=0)
    target_weights = avg_attn[target_indices]
    max_target_weight = np.max(target_weights)
    return max_target_weight
def encode_input_text(text, tokenizer):
    tokenized_text = tokenizer.tokenize(text)
    #ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenized_text))
    ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    return tokenizer.build_inputs_with_special_tokens(ids)


class GPT2_collator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [int(sequence['label']) for sequence in sequences]
        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels': torch.tensor(labels)})
        return inputs


config_class = GPT2Config
model_class = GPT2LMHeadModel
tokenizer_class = GPT2Tokenizer

max_len = 1000
batch_size = 8
epochs = 2
num_labels = 60
K_number = 4
TH_value = 0.9
Split_number = 5
Model_name = 'gpt2'

print('Loading gpt-2 model')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=Model_name, num_labels=num_labels, output_attentions=True)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=Model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=Model_name, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model = nn.DataParallel(model)
model.to(device)



def pre_process(dataset):
    dataset['clean_tweet'] = dataset['text']#.apply(lambda x: x.replace('#', ' '))
    return dataset

class DatasetCreator(Dataset):
    def __init__(self, processed_data, train):
        self.data = processed_data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data.iloc[index]
        if self.train:
            return {'text': line['text'], 'label': line['target']}
        else:
            return {'text': line['text'], 'label': 0}

def get_labels(file):
    df = pd.read_csv(file)
    return np.array(df['target'])

def prep_array(array):
    intArray1 = np.array([])
    for value in array:
        try:
            intArray1 = np.append(intArray1, [int(value, 16)])
        except:
            continue
    return intArray1

# Function for training
def train(dataloader, optimizer, scheduler, device):
    global model
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
            loss = loss.mean()  # Manual reduction to scalar if necessary

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

# Function for validation
def validate(dataloader, device):
    global model
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
                loss = loss.mean()  # Manual reduction to scalar if necessary

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
        #newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            if ind == 0:
                predictions_labels = logits.to('cpu').numpy()
            else:
                predictions_labels = np.concatenate((predictions_labels, logits.to('cpu').numpy()), axis=0)

            #predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    return predictions_labels

def predict_mod(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    y_labels = []

    for ind,d_batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch = {k:v.type(torch.long).to(device) for k,v in d_batch.items()}
        for k,v in d_batch.items():
            print(d_batch.items())
            sys.exit()
            y_labels.append(int(v.to('cpu')))
        #newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            batch.to('cpu')
            if ind == 0:
                predictions_labels = logits.to('cpu').numpy()
                #y_labels = np.array(list(batch.values()))
            else:
                predictions_labels = np.concatenate((predictions_labels, logits.to('cpu').numpy()), axis=0)
                #y_labels = np.concatenate((y_labels, np.array(list(batch.values()))), axis = 0)
            #predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    return predictions_labels, np.array(y_labels)

dataset = pd.read_csv('/mnt/sdrive/yasod/TF_ET-BERT/ET-BERT/datasets/DF/rearranged_data/train.csv')
val_dataset = pd.read_csv('/mnt/sdrive/yasod/TF_ET-BERT/ET-BERT/datasets/DF/rearranged_data/valid.csv')

gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=max_len)

# Prepare training data
train_data = DatasetCreator(processed_data, train=True)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

# Prepare validation data
val_data = DatasetCreator(val_processed, train=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8, weight_decay=0.01)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
loss = []
accuracy = []
val_loss_list = []
val_accuracy_list = []

for epoch in tqdm(range(epochs)):
    train_labels, true_labels, train_loss = train(train_dataloader, optimizer, scheduler, device)
    train_acc = accuracy_score(true_labels, train_labels)
    print('epoch: %.2f train accuracy %.2f' % (epoch, train_acc))
    loss.append(train_loss)
    accuracy.append(train_acc)

    val_labels, val_true_labels, val_loss = validate(val_dataloader, device)
    val_acc= accuracy_score(val_true_labels, val_labels)
    print('epoch: %.2f validation accuracy %.2f' % (epoch, val_acc))
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_acc)

# torch.save(model.state_dict(),'/mnt/sdrive/yasod/TF_ET-BERT/ET-BERT/gpt2_approach/models/trained_gpt_DF.pth', )
#model.load_state_dict(torch.load('/mnt/sdrive/yasod/TF_ET-BERT/ET-BERT/gpt2_approach/models/trained_gpt_IoT.pth'))
model.eval()

random_classes = random.sample(range(num_labels), 4)
test_data_cont = 100
test_batch_size = 10

test_dataset = pd.read_csv('/mnt/sdrive/yasod/TF_ET-BERT/ET-BERT/datasets/IoT/rearranged_data/test.csv')
test_dataset = test_dataset[test_dataset['target'].isin(random_classes)].reset_index(drop=True)
test_dataset = test_dataset.iloc[:test_data_cont, :]

test_data = DatasetCreator(test_processed, train=True)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, collate_fn=gpt2_collator)

attention_outputs = []
def get_attention_output(module, input, output):
    attention_outputs.append(output[-1])

last_attention_layer = model.module.transformer.h[-1].attn
hook = last_attention_layer.register_forward_hook(get_attention_output)

output = predict(test_dataloader, device)
hook.remove()

for i in range(2* math.ceil(test_data_cont//test_batch_size)):
    if i == 0:
        merged_attention_tensors = attention_outputs[i].detach().to('cpu').numpy()
    else:
        merged_attention_tensors = np.concatenate((merged_attention_tensors,attention_outputs[i].detach().to('cpu').numpy()), axis = 0)

resultant_tensor = merged_attention_tensors.sum(axis=1)

square_value = 500
data1 = []
data2 = []
data3 = []
data4 = []

values1 = []
values2 = []
values3 = []
values4 = []

for i in range(len(resultant_tensor)):
    if test_dataset['target'][i] == random_classes[0]:
        data1.append(resultant_tensor[i])
        values1.append(prep_array(test_dataset['text'][i].split(' ')))

    elif test_dataset['target'][i] == random_classes[1]:
        data2.append(resultant_tensor[i])
        values2.append(prep_array(test_dataset['text'][i].split(' ')))

    elif test_dataset['target'][i] == random_classes[2]:
        data3.append(resultant_tensor[i])
        values3.append(prep_array(test_dataset['text'][i].split(' ')))

    elif test_dataset['target'][i] == random_classes[3]:
        data4.append(resultant_tensor[i])
        values4.append(prep_array(test_dataset['text'][i].split(' ')))

data1 = np.average(np.array(data1),axis=0)*255
data2 = np.average(np.array(data2),axis=0)*255
data3 = np.average(np.array(data3),axis=0)*255
data4 = np.average(np.array(data4),axis=0)*255

values1 = np.average(np.array(values1).astype(int),axis=0)
values2 = np.average(np.array(values2).astype(int),axis=0)
values3 = np.average(np.array(values3).astype(int),axis=0)
values4 = np.average(np.array(values4).astype(int),axis=0)


new_array1 = [[0]* square_value] * square_value
new_array1 = np.array(new_array1)


for i in range(len(data1)):
  for j in range(len(data1[i])):
    if i<square_value and j<square_value:
        new_array1[i][j] = data1[i][j] + 50
        if new_array1[i][j] > 255:
            new_array1[i][j] = 255.0

new_array2 = [[0]* square_value] * square_value
new_array2 = np.array(new_array1)


for i in range(len(data2)):
  for j in range(len(data2[i])):
    if i<square_value and j<square_value:
        new_array2[i][j] = data2[i][j] + 100
        if new_array2[i][j] > 255:
            new_array2[i][j] = 255.0

new_array3 = [[0]* square_value] * square_value
new_array3 = np.array(new_array1)

for i in range(len(data3)):
  for j in range(len(data3[i])):
    if i<square_value and j<square_value:
        new_array3[i][j] = data3[i][j] + 100
        if new_array3[i][j] > 255:
            new_array3[i][j] = 255.0

new_array4 = [[0]* square_value] * square_value
new_array4 = np.array(new_array4)

for i in range(len(data4)):
  for j in range(len(data4[i])):
    if i<square_value and j<square_value:
        new_array4[i][j] = data4[i][j] + 100
        if new_array4[i][j] > 255:
            new_array4[i][j] = 255.0


intArray1 = values1
intArray2 = values2
intArray3 = values3
intArray4 = values4

plt.figure(figsize=(16, 6)) 

plt.subplot(2, 4, 1) 
plt.plot(intArray1[:square_value])
plt.title('Class ' + str(random_classes[0]))
plt.xlabel('Trace')
plt.ylabel('Value')


plt.subplot(2, 4, 5)  
plt.imshow(new_array1, cmap='viridis')
plt.title('')
plt.xlabel('Trace')
plt.ylabel('Trace')


plt.subplot(2, 4, 2)
plt.plot(intArray2[:square_value])
plt.title('Class ' + str(random_classes[1]))
plt.xlabel('Trace')
plt.ylabel('Value')


plt.subplot(2, 4, 6)
plt.imshow(new_array2, cmap='viridis')
plt.title('')
plt.xlabel('Trace')
plt.ylabel('Trace')



plt.subplot(2, 4, 3)
plt.plot(intArray3[:square_value])
plt.title('Class ' + str(random_classes[2]))
plt.xlabel('Trace')
plt.ylabel('Value')

plt.subplot(2, 4, 7) 
plt.imshow(new_array3, cmap='viridis')
plt.title('')
plt.xlabel('Trace')
plt.ylabel('Trace')


plt.subplot(2, 4, 4)
plt.plot(intArray4[:square_value])
plt.title('Class ' + str(random_classes[3]))
plt.xlabel('Trace')
plt.ylabel('Value')

plt.subplot(2, 4, 8)
plt.imshow(new_array4, cmap='viridis')
plt.title('')
plt.xlabel('Trace')
plt.ylabel('Trace')

plt.tight_layout()  
plt.savefig('attention_visualization_AWF.pdf', format='pdf')  
plt.show()


