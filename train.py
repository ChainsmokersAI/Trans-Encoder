import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

class BiEncoderDataset(Dataset):
    """
    Dataset for Training Bi-Encoder
    """
    def __init__(self, path_dataset, tokenizer, max_sent_len=512):
        # Data: Sentence Pair
        self.sent0=[]
        self.sent1=[]
        # Label
        self.label=[]
        
        # Read Dataset
        df=pd.read_csv(path_dataset)
        
        for idx in df.index:
            row=df.loc[idx]
            
            # Encode Sentence
            enc0=tokenizer.encode(row["sent0"], truncation=True, max_length=max_sent_len)
            enc1=tokenizer.encode(row["sent1"], truncation=True, max_length=max_sent_len)
            
            # Append Data
            self.sent0.append(enc0)
            self.sent1.append(enc1)
            # Append Label
            self.label.append(float(row["pseudo_label"]))
            
        print(len(self.sent0), "data processed")
            
    def __getitem__(self, idx):
        return self.sent0[idx], self.sent1[idx], self.label[idx]
    
    def __len__(self):
        return len(self.sent0)

class CrossEncoderDataset(Dataset):
    """
    Dataset for Training Cross-Encoder
    """
    def __init__(self, path_dataset, tokenizer, max_sent_len=256):
        self.data=[]
        self.label=[]
        
        # Read Dataset
        df=pd.read_csv(path_dataset)
        
        for idx in df.index:
            row=df.loc[idx]
            
            # Encode Sentence
            enc0=tokenizer.encode(row["sent0"], truncation=True, max_length=max_sent_len)
            enc1=tokenizer.encode(row["sent1"], truncation=True, max_length=max_sent_len)
            
            # Append Data
            self.data.append(enc0[:-1]+[tokenizer.sep_token_id]+enc1[1:])
            self.label.append(float(row["pseudo_label"]))
            
        print(len(self.data), "data proceesed")
            
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)

def get_bi_collate_fn(tokenizer):
    """
    Return Collate Function for Bi-Encoder Batch
    """
    def collate_fn(batch):
        # Max Sentence Length
        max_seq_len0=0
        max_seq_len1=0
        for sent0, sent1, _ in batch:
            if len(sent0)>max_seq_len0: max_seq_len0=len(sent0)
            if len(sent1)>max_seq_len1: max_seq_len1=len(sent1)

        # Data: Sentence Pair
        batch_sent0=[]
        batch_sent1=[]
        # Label
        batch_label=[]
        for sent0, sent1, label in batch:
            sent0.extend([tokenizer.pad_token_id]*(max_seq_len0-len(sent0)))
            batch_sent0.append(sent0)
            
            sent1.extend([tokenizer.pad_token_id]*(max_seq_len1-len(sent1)))
            batch_sent1.append(sent1)

            batch_label.append(label)

        return torch.tensor(batch_sent0), torch.tensor(batch_sent1), torch.tensor(batch_label)
    
    return collate_fn

def get_cross_collate_fn(tokenizer):
    """
    Return Collate Function for Cross-Encoder Batch
    """
    def collate_fn(batch):
        # Max Sentence Length
        max_seq_len=0
        for data, _ in batch:
            if len(data)>max_seq_len: max_seq_len=len(data)

        batch_data=[]
        batch_label=[]
        for data, label in batch:
            data.extend([tokenizer.pad_token_id]*(max_seq_len-len(data)))
            batch_data.append(data)

            batch_label.append(label)

        return torch.tensor(batch_data), torch.tensor(batch_label)
    
    return collate_fn

def train(
    type,
    n_loop,
    path_dataset,
    tokenizer,
    model,
    # Hyperparams
    batch_size,
    accum_steps,
    lr,
    epochs,
    loss_func
):
    """
    Train LM with Pseudo-Labels
    """
    # Pseudo-Labeled Dataset
    if type=="bi2cross":
        dataset_train=CrossEncoderDataset(path_dataset=path_dataset, tokenizer=tokenizer)
        collate_fn=get_cross_collate_fn(tokenizer=tokenizer)
    elif type=="cross2bi":
        dataset_train=BiEncoderDataset(path_dataset=path_dataset, tokenizer=tokenizer)
        collate_fn=get_bi_collate_fn(tokenizer=tokenizer)
    # DataLoader
    dataloader_train=DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Loss: MSE
    if loss_func=="MSE":
        train_loss=nn.MSELoss()
    # Loss: BCE
    elif loss_func=="BCE":
        train_loss=nn.BCEWithLogitsLoss()
        
    # Optimizer, Scheduler
    optimizer=AdamW(model.parameters(), lr=lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.05*epochs*len(dataset_train)/(accum_steps*batch_size)),
        num_training_steps=int(epochs*len(dataset_train)/(accum_steps*batch_size))
    )

    # Mixed Precision: GradScaler
    scaler=amp.GradScaler()

    # Tensorboard
    writer=SummaryWriter()
    
    step_global=0
    for epoch in range(epochs):
        _loss=0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader_train):
            # Load Data, Label
            sent=batch[0].to(model.pretrained.device)
            label=batch[-1].to(model.pretrained.device)
            # cross2bi
            if len(batch)==3:
                sent2=batch[1].to(model.pretrained.device)

            # Forward
            with amp.autocast():
                if type=="bi2cross":
                    pred=model(sent)
                elif type=="cross2bi":
                    pred=model(sent, sent2)
                loss=train_loss(pred, label.unsqueeze(-1))/accum_steps
            # Backward
            scaler.scale(loss).backward()
            _loss+=loss.item()

            # Step
            if (step+1)%accum_steps==0:
                step_global+=1

                # Tensorboard
                if type=="bi2cross":
                    name_target=f'cross-encoder_distilled_loop{n_loop}_epoch{epochs}'
                elif type=="cross2bi":
                    name_target=f'bi-encoder_distilled_loop{n_loop}_epoch{epochs}'
                writer.add_scalar(
                    name_target,
                    _loss,
                    step_global
                )
                _loss=0

                # Optimizer, Scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        # Save Model
        _device=model.pretrained.device
        model.to(torch.device("cpu"))

        if type=="bi2cross":
            path_target=f'../model/cross-encoder_distilled_loop{n_loop}_epoch{epoch+1}of{epochs}.pth'
        elif type=="cross2bi":
            path_target=f'../model/bi-encoder_distilled_loop{n_loop}_epoch{epoch+1}of{epochs}.pth'
            
        torch.save(
            model.state_dict(),
            path_target
        )

        model.to(_device)

    return model
