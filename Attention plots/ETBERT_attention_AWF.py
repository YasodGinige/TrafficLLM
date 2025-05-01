"""
This script provides an exmaple to wrap UER-py for classification.
"""
import sys
import random
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts, infer_opts
from uer.model_loader import load_model
from sklearn.metrics import accuracy_score
import tqdm
import numpy as np
import gc
import pandas as pd
from uer.encoders.transformer_encoder import TransformerEncoder
import matplotlib.pyplot as plt

Split_Number = 5
result_file = 'results_DC_etbert_unenc.txt'

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output, attention_weights = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                logits = logits[:len(tgt)]
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits, attention_weights
        else:
            return None, logits, attention_weights
            #return temp_output, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    df = pd.read_csv(path)
    for i in df['label']:
        labels_set.add(i)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        print('##################', args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    df = pd.read_csv(path)
    for line_id, line in enumerate(df['text_a']):
        
        line = line
        tgt = int(df['label'][line_id])
        if args.soft_targets and "logits" in columns.keys():
            soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
        #if "text_b" not in columns:  # Sentence classification.
        text_a = line
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
        seg = [1] * len(src)
        # else:  # Sentence-pair classification.
        #     text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
        #     src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
        #     src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
        #     src = src_a + src_b
        #     seg = [1] * len(src_a) + [2] * len(src_b)

        if len(src) > args.seq_length:
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)
        if args.soft_targets and "logits" in columns.keys():
            dataset.append((src, tgt, seg, soft_tgt))
        else:
            dataset.append((src, tgt, seg))

    return dataset


# ============================================================================================================== #

def claculate_mean_vectors(NB_CLASSES, model_predictions, y_train):
    for i in range(NB_CLASSES):
        variable_name = f"Mean_{i}"
        locals()[variable_name]=np.array([0] * NB_CLASSES)

    #model_predictions=args.model.predict(X_train)

    count=[0]*NB_CLASSES
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)

    for i in range(len(model_predictions)):
        k=np.argmax(model_predictions[i])
        if (np.argmax(model_predictions[i])==y_train[i]):
            Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])] + model_predictions[i]
            count[y_train[i]]+=1
    #print("Counts: ",count)

    Mean_Vectors=[]
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])

    Mean_Vectors=np.array(Mean_Vectors)
    return Mean_Vectors

def calculate_thresholds(NB_CLASSES, model_predictions, y_valid, Mean_Vectors, K_number):
    #model_predictions = args.model.predict(X_valid)
    TH_value = 0.9

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

    return Threasholds_1, Threasholds_2, Threasholds_3, Indexes


def print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES):

    y_test = y_test.astype(int)
    y_open = y_open.astype(int)

    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    print('Test accuracy Normal model_Closed_set :', acc_Close)

    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy Normal model_Open_set :', acc_Open)

    y_test=y_test[:len(prediction_classes)]
    y_open=y_open[:len(prediction_classes_open)]

    Matrix=[]
    for i in range(NB_CLASSES+1):
        Matrix.append(np.zeros(NB_CLASSES+1))

    for i in range(len(y_test)):
        Matrix[y_test[i]][prediction_classes[i]]+=1

    for i in range(len(y_open)):
        Matrix[y_open[i]][prediction_classes_open[i]]+=1

    F1_Score_new=New_F1_Score(Matrix, NB_CLASSES)
    print("Average novel F1_Score: ", F1_Score_new)
    print("\n", "Micro")

    F1_Score_micro=Micro_F1(Matrix, NB_CLASSES)
    print("Average Micro F1_Score: ", F1_Score_micro)

    print("\n", "Macro")
    F1_Score_macro=Macro_F1(Matrix, NB_CLASSES)
    print("Average Macro F1_Score: ", F1_Score_macro)

    text_file = open("./etbert_results/"+ result_file, "a")

    text_file.write('Test accuracy Normal model_Closed_set :'+ str(acc_Close) + '\n')
    text_file.write('Test accuracy Normal model_Open_set :'+ str(acc_Open) + '\n')
    text_file.write("Average novel F1_Score: " + str(F1_Score_new) + '\n')
    text_file.write("Average Micro F1_Score: " + str(F1_Score_micro) + '\n')
    text_file.write("Average Macro F1_Score: " + str(F1_Score_macro) + '\n')
    text_file.write('\n')
    for p in Matrix:
        text_file.write( np.array_str(p) + '\n')
    
    text_file.write('\n')
    text_file.write('\n')
    text_file.close()


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

    prediction_classes=np.array(prediction_classes)

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


def Micro_F1(Matrix, NB_CLASSES):
    epsilon = 1e-8
    TP = 0
    FP = 0
    TN = 0

    for k in range(NB_CLASSES):
        TP += Matrix[k][k]
        FP += (np.sum(Matrix, axis=0)[k] - Matrix[k][k])
        TN += (np.sum(Matrix, axis=1)[k] - Matrix[k][k])

    Micro_Prec = TP / (TP + FP)
    Micro_Rec = TP / (TP + TN)
    print("Micro_Prec:", Micro_Prec)
    print("Micro_Rec:", Micro_Rec)
    Micro_F1 = 2 * Micro_Prec * Micro_Rec / (Micro_Rec + Micro_Prec + epsilon)

    return Micro_F1


def New_F1_Score(Matrix, NB_CLASSES):
    Column_sum = np.sum(Matrix, axis=0)
    Raw_sum = np.sum(Matrix, axis=1)

    Precision_Differences = []
    Recall_Differences = []
    for i in range(NB_CLASSES):
        Precision_Differences.append(np.abs(2 * Matrix[i][i] - Column_sum[i]))
        Recall_Differences.append(np.abs(2 * Matrix[i][i] - Raw_sum[i]))

    Precision_Differences = np.array(Precision_Differences)
    Precision_Differences_Per = Precision_Differences / np.sum(Precision_Differences)
    Recall_Differences = np.array(Recall_Differences)
    Recall_Differences_Per = Recall_Differences / np.sum(Recall_Differences)

    Precisions = np.zeros(NB_CLASSES)
    Recalls = np.zeros(NB_CLASSES)

    epsilon = 1e-8
    for k in range(len(Precisions)):
        Precisions[k] = (Matrix[k][k] / np.sum(Matrix, axis=0)[k])
    Precision = np.sum(np.array(Precisions) * Precision_Differences_Per)

    for k in range(len(Recalls)):
        Recalls[k] = (Matrix[k][k] / np.sum(Matrix, axis=1)[k])  # *Recall_Differences_Per[k]
    Recall = np.sum(np.array(Recalls) * Recall_Differences_Per)

    print("Precision:", Precision)
    print("Recall:", Recall)

    F1_Score = 2 * Precision * Recall / (Precision + Recall + epsilon)
    return F1_Score


def Macro_F1(Matrix, NB_CLASSES):
    Precisions = np.zeros(NB_CLASSES)
    Recalls = np.zeros(NB_CLASSES)

    epsilon = 1e-8

    for k in range(len(Precisions)):
        Precisions[k] = Matrix[k][k] / np.sum(Matrix, axis=0)[k]
    
    Precision = np.average(Precisions)
    for k in range(len(Recalls)):
        Recalls[k] = Matrix[k][k] / np.sum(Matrix, axis=1)[k]

    Recall = np.average(Recalls)
    print("Macro Prec:", Precision)
    print("Macro Rec:", Recall)

    F1_Score = 2 * Precision * Recall / (Precision + Recall + epsilon)
    return F1_Score

# ===================================================================================================== #

    

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _, attention_weights = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss

def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits, attention_weights = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        with open("/data2/lxj/pre-train/results/confusion_matrix",'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion

def prep_array(array):
    intArray1 = np.array([])
    for value in array:
        try:
            intArray1 = np.append(intArray1, [int(value, 16)])
        except:
            continue
    return intArray1

def filterd_dataset(args, data_path, num_labels, random_classes):
    
    dataset, columns = [], {}
    df = pd.read_csv(data_path)
    plain_text = []
    for line_id, line in enumerate(df['text_a']):
        line = line
        tgt = int(df['label'][line_id])
        if len(dataset) > 120:
            break
        if tgt in random_classes:
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            #if "text_b" not in columns:  # Sentence classification.
            text_a = line
            plain_text.append(text_a)
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))
    dataset = dataset[:100]
    plain_text = plain_text[:100]

    return dataset, plain_text



def read_dataset_USTC(args, path, label_path):
    dataset, columns = [], {}
    f = np.load(path)
    labels = np.load(label_path)

    for line_id, line in enumerate(f):

        tgt = int(labels[line_id])
        if args.soft_targets and "logits" in columns.keys():
            soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
        #if "text_b" not in columns:  # Sentence classification.
        text_a = line
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
        seg = [1] * len(src)

        if len(src) > args.seq_length:
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)
        if args.soft_targets and "logits" in columns.keys():
            dataset.append((src, tgt, seg, soft_tgt))
        else:
            dataset.append((src, tgt, seg))

    return dataset



def rearrange_array(data1, square_value):
    new_array1 = [[0]* square_value] * square_value
    new_array1 = np.array(new_array1)

    for i in range(len(data1)):
        for j in range(len(data1[i])):
            if i<square_value and j<square_value:
                new_array1[i][j] = data1[i][j] + 50
                if new_array1[i][j] > 255:
                    new_array1[i][j] = 255.0
    return new_array1

def plot_attention_open(resultant_tensor,random_classes, filename, labels, plain_text):
    square_value = 500
    selection = random.sample(range(100), 4)
    resultant_tensor = resultant_tensor.sum(axis=1)*255.0

    plt.figure(figsize=(16, 6))  # Adjust the figure size as needed

    # Plot for intArray
    plt.subplot(2, 4, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(prep_array(plain_text[selection[0]][:square_value].split(' '))) 
    plt.title('Class ' + str(selection[0]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 5)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(rearrange_array(resultant_tensor[selection[0]], square_value), cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')


    plt.subplot(2, 4, 2)  # 2 rows, 1 column, 1st subplot
    plt.plot(prep_array(plain_text[selection[1]][:square_value].split(' ')))
    plt.title('Class ' + str(selection[1]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 6)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(rearrange_array(resultant_tensor[selection[1]], square_value), cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')

    plt.subplot(2, 4, 3)  # 2 rows, 1 column, 1st subplot
    plt.plot(prep_array(plain_text[selection[2]][:square_value].split(' ')))
    plt.title('Class ' + str(selection[2]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 7)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(rearrange_array(resultant_tensor[selection[2]], square_value), cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')

    plt.subplot(2, 4, 4)  # 2 rows, 1 column, 1st subplot
    plt.plot(prep_array(plain_text[selection[3]][:square_value].split(' ')))
    plt.title('Class ' + str(selection[3]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 8)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(rearrange_array(resultant_tensor[selection[3]], square_value), cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')

    # Display the plots
    plt.tight_layout()
    plt.savefig(filename, format='pdf')  # Save as PDF file
    plt.show()
    


def plot_attention(resultant_tensor,random_classes, filename, labels, plain_text):
    square_value = 500
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    values1 = []
    values2 = []
    values3 = []
    values4 = []

    print('random_classes:', random_classes)
    print('resultant_tensor',resultant_tensor.shape)
    resultant_tensor = resultant_tensor.sum(axis=1)
    print('resultant_tensor',resultant_tensor.shape)

    for i in range(len(resultant_tensor)):
        print(labels[i], random_classes)
        if labels[i] == random_classes[0]:
            data1.append(resultant_tensor[i])
            temp_array = prep_array(plain_text[i].split(' '))
            if len(temp_array) > square_value:
                values1.append(temp_array[:square_value])

        elif labels[i] == random_classes[1]:
            data2.append(resultant_tensor[i])
            temp_array = prep_array(plain_text[i].split(' '))
            if len(temp_array) > square_value:
                values2.append(temp_array[:square_value])

        elif labels[i] == random_classes[2]:
            data3.append(resultant_tensor[i])
            temp_array = prep_array(plain_text[i].split(' '))
            if len(temp_array) > square_value:
                values3.append(temp_array[:square_value])

        elif labels[i] == random_classes[3]:
            data4.append(resultant_tensor[i])
            temp_array = prep_array(plain_text[i].split(' '))
            if len(temp_array) > square_value:
                values4.append(temp_array[:square_value])

    print('data1',np.array(data1).shape)
    data1 = np.exp(np.average(np.array(data1),axis=0))*125
    data2 = np.exp(np.average(np.array(data2),axis=0))*125
    data3 = np.exp(np.average(np.array(data3),axis=0))*125
    data4 = np.exp(np.average(np.array(data4),axis=0))*125
    print('data1',data1.shape)


    print('values1',np.array(values1).shape)
    values1 = np.average(np.array(values1),axis=0)
    values2 = np.average(np.array(values2),axis=0)
    values3 = np.average(np.array(values3),axis=0)
    values4 = np.average(np.array(values4),axis=0)
    print('values1',values1.shape)

    new_array1 = [[0]* square_value] * square_value
    new_array1 = np.array(new_array1)

    print(values1.shape)
    print(data1.shape)

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
                new_array2[i][j] = data2[i][j] + 50
                if new_array2[i][j] > 255:
                    new_array2[i][j] = 255.0

    new_array3 = [[0]* square_value] * square_value
    new_array3 = np.array(new_array1)

    for i in range(len(data3)):
        for j in range(len(data3[i])):
            if i<square_value and j<square_value:
                new_array3[i][j] = data3[i][j] + 50
                if new_array3[i][j] > 255:
                    new_array3[i][j] = 255.0

    new_array4 = [[0]* square_value] * square_value
    new_array4 = np.array(new_array4)

    for i in range(len(data4)):
        for j in range(len(data4[i])):
            if i<square_value and j<square_value:
                new_array4[i][j] = data4[i][j] + 50
                if new_array4[i][j] > 255:
                    new_array4[i][j] = 255.0


    intArray1 = values1
    intArray2 = values2
    intArray3 = values3
    intArray4 = values4

    plt.figure(figsize=(16, 6))  # Adjust the figure size as needed

    # Plot for intArray
    plt.subplot(2, 4, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(intArray1)
    plt.title('Class ' + str(random_classes[0]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 5)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(new_array1, cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')


    plt.subplot(2, 4, 2)  # 2 rows, 1 column, 1st subplot
    plt.plot(intArray2)
    plt.title('Class ' + str(random_classes[1]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 6)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(new_array2, cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')

    plt.subplot(2, 4, 3)  # 2 rows, 1 column, 1st subplot
    plt.plot(intArray3)
    plt.title('Class ' + str(random_classes[2]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 7)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(new_array3, cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')

    plt.subplot(2, 4, 4)  # 2 rows, 1 column, 1st subplot
    plt.plot(intArray4)
    plt.title('Class ' + str(random_classes[3]))
    plt.xlabel('Trace')
    plt.ylabel('Value')

    # Plot for data
    plt.subplot(2, 4, 8)  # 2 rows, 1 column, 2nd subplot
    plt.imshow(new_array4, cmap='magma_r')
    plt.title('')
    plt.xlabel('Trace')
    plt.ylabel('Trace')

    # Display the plots
    plt.tight_layout()  # Automatically adjusts subplot params to give specified padding
    plt.savefig(filename, format='pdf')  # Save as PDF file
    plt.show()

def visualize_attention(args, dataset, openset, random_classes, random_classes_open, close_set_filename, open_set_filename, plain_text, plain_text_open):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    src_open = torch.LongTensor([example[0] for example in openset])
    tgt_open = torch.LongTensor([example[1] for example in openset])
    seg_open = torch.LongTensor([example[2] for example in openset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    args.model.eval()

    for i, (src_batch_test, tgt_batch_test, seg_batch_test, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch_test = src_batch_test.to(args.device)
        tgt_batch_test = tgt_batch_test.to(args.device)
        seg_batch_test = seg_batch_test.to(args.device)
        with torch.no_grad():
            _, logit_test, attention_weights = args.model(src_batch_test, tgt_batch_test, seg_batch_test)
        
        if i == 0:
            attention_weights_bucket = attention_weights.cpu().detach().numpy()
            test_labels = tgt_batch_test.cpu().detach().numpy()
            texts = src_batch_test.cpu().detach().numpy()
        else:
            attention_weights_bucket = np.concatenate((attention_weights_bucket,attention_weights.cpu().detach().numpy()),axis=0)
            test_labels = np.concatenate((test_labels,tgt_batch_test.cpu().detach().numpy()),axis=0)
            texts = np.concatenate((texts,src_batch_test.cpu().detach().numpy()),axis=0)

    del src_batch_test,tgt_batch_test,seg_batch_test,src, tgt, seg
    gc.collect()
    torch.cuda.empty_cache()
    
    for i, (src_batch_open, tgt_batch_open, seg_batch_open, _) in enumerate(batch_loader(batch_size, src_open, tgt_open, seg_open)):
        src_batch_open = src_batch_open.to(args.device)
        tgt_batch_open = tgt_batch_open.to(args.device)
        seg_batch_open = seg_batch_open.to(args.device)
        with torch.no_grad():
            _, logit_open, attention_weights = args.model( src = src_batch_open, tgt = None, seg = seg_batch_open)
        if i == 0:
            attention_weights_bucket_open = attention_weights.cpu().detach().numpy()
            open_labels = np.ones(tgt_batch_open.size()) * args.labels_num
            texts_open = src_batch_open.cpu().detach().numpy()
        else:
            attention_weights_bucket_open = np.concatenate((attention_weights_bucket_open,attention_weights.cpu().detach().numpy()),axis=0)
            open_labels = np.concatenate((open_labels,tgt_batch_open.cpu().detach().numpy()),axis=0)
            texts_open = np.concatenate((texts_open,src_batch_open.cpu().detach().numpy()),axis=0)

    plot_attention(attention_weights_bucket,random_classes, close_set_filename, test_labels, plain_text)
    plot_attention(attention_weights_bucket_open,random_classes_open, open_set_filename, open_labels, plain_text_open)


def evaluate_mod(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])


    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()
    K_number = 4

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src_train, tgt_train, seg_train)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logit, attention_weights = args.model(src_batch, tgt_batch, seg_batch)

        # Here we implement our method for the general and openset classification   ##########################
        if i == 0:
            train_data_bucket = logit.cpu().detach().numpy()
            train_labels =tgt_batch.cpu().detach().numpy()
        elif i < int(len(tgt_train)*0.6/batch_size):
            train_data_bucket = np.concatenate((train_data_bucket,logit.cpu().detach().numpy()),axis=0)
            train_labels = np.concatenate((train_labels,tgt_batch.cpu().detach().numpy()),axis=0)

        elif i == int(len(tgt_train)*0.6/batch_size):
            valid_data_bucket = logit.cpu().detach().numpy()
            valid_labels =tgt_batch.cpu().detach().numpy()
        
        elif i > int(len(tgt_train)*0.6/batch_size):
            valid_data_bucket = np.concatenate((valid_data_bucket,logit.cpu().detach().numpy()),axis=0)
            valid_labels = np.concatenate((valid_labels,tgt_batch.cpu().detach().numpy()),axis=0)



        pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
    

    for i, (src_batch_test, tgt_batch_test, seg_batch_test, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch_test = src_batch_test.to(args.device)
        tgt_batch_test = tgt_batch_test.to(args.device)
        seg_batch_test = seg_batch_test.to(args.device)
        with torch.no_grad():
            _, logit_test, attention_weights = args.model(src_batch_test, tgt_batch_test, seg_batch_test)

        if i == 0:
            test_data_bucket = logit_test.cpu().detach().numpy()
            test_labels = tgt_batch_test.cpu().detach().numpy()
        else:
            test_data_bucket = np.concatenate((test_data_bucket,logit_test.cpu().detach().numpy()),axis=0)
            test_labels = np.concatenate((test_labels,tgt_batch_test.cpu().detach().numpy()),axis=0)
    

    torch.cuda.empty_cache()
    for i, (src_batch_open, tgt_batch_open, seg_batch_open, _) in enumerate(batch_loader(batch_size, src_open, tgt_train[:len(src_open)], seg_open)):
    
        src_batch_open = src_batch_open.to(args.device)
        tgt_batch_open = tgt_batch_open.to(args.device)
        seg_batch_open = seg_batch_open.to(args.device)
        with torch.no_grad():
            _, logit_open, attention_weights = args.model(src_batch_open, tgt_batch_open, seg_batch_open)

        if i == 0:
            open_data_bucket = logit_open.cpu().detach().numpy()
            open_labels = np.ones(tgt_batch_open.size()) * args.labels_num
        else:
            open_data_bucket = np.concatenate((open_data_bucket,logit_open.cpu().detach().numpy()),axis=0)
            open_labels = np.concatenate((open_labels,np.ones(tgt_batch_open.size()) * args.labels_num),axis=0)

    Mean_Vectors = claculate_mean_vectors(args.labels_num, train_data_bucket, train_labels)
    Threasholds_1, Threasholds_2, Threasholds_3, Indexes = calculate_thresholds(args.labels_num, valid_data_bucket, valid_labels, Mean_Vectors, K_number)
    final_classification(args.labels_num, test_data_bucket, open_data_bucket, test_labels, open_labels, Mean_Vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3)

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        # with open("/data2/lxj/pre-train/results/confusion_matrix",'w') as f:
        #     for cf_a in cf_array:
        #         f.write(str(cf_a)+'\n')
        # print("Report precision, recall, and f1:")
        # eps = 1e-9
        # for i in range(confusion.size()[0]):
        #     p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        #     r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        #     if (p + r) == 0:
        #         f1 = 0
        #     else:
        #         f1 = 2 * p * r / (p + r)
        #     print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)
    #infer_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")



    ################  training phase ##########################

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(7)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    
    global src_train, tgt_train, seg_train
    src_train = torch.LongTensor([example[0] for example in trainset])
    tgt_train = torch.LongTensor([example[1] for example in trainset])
    seg_train = torch.LongTensor([example[2] for example in trainset])
    

    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    # for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
    #     model.train()
    #     for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src_train, tgt_train, seg_train, soft_tgt)):
    #         loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    #         total_loss += loss.item()
    #         if (i + 1) % args.report_steps == 0:
    #             print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
    #             total_loss = 0.0

    #     result = evaluate(args, read_dataset(args, args.dev_path))
    #     if result[0] > best_result:
    #         best_result = result[0]
    #         save_model(model, args.output_model_path)

    
    # Evaluation phase.
    if args.test_path is not None:
        model.module.load_state_dict(torch.load(args.output_model_path))
        print("Test set evaluation.")
        close_set_filename = './etbert_attention/etbert_attention_AWF.pdf'
        open_set_filename = './etbert_attention/etbert_attention_AWF_open.pdf'

        random_classes_close = random.sample(range(60, args.labels_num), 4)
        random_classes_open = random.sample(range(args.labels_num, 200), 4)

        dataset, plain_text = filterd_dataset(args, args.test_path, args.labels_num, random_classes_close)
        openset, plain_text_open = filterd_dataset(args, args.open_path, args.labels_num, random_classes_open)

        visualize_attention(args, dataset, openset, random_classes_close, random_classes_open, close_set_filename, open_set_filename, plain_text, plain_text_open)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
