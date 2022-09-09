import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

from models import BiEncoder, CrossEncoder
from evaluation import evaluate

def pseudo_label(
    type, # Model Type: bi2cross or cross2bi
    n_loop,
    tokenizer,
    model,
    path_dataset="../dataset/stsbenchmark/sts-train.csv",
    method="same"
):
    """
    Pseudo-Labeling using Pre-Trained (Distilled) LM
    """
    ## 1. Read Dataset
    # Raw Dataset
    if "stsbenchmark" in path_dataset:
        data=open(path_dataset).read().split("\n")
        data.remove("")
        
        sents0=[]
        sents1=[]
        # Parse Dataset
        for _data in data:
            sent0, sent1=_data.split("\t")[5:7]
            
            sents0.append(sent0)
            sents1.append(sent1)
            
        # Make DataFrame
        df_pseudo=pd.DataFrame({
            "sent0": sents0,
            "sent1": sents1,
        })
    else:
        df_pseudo=pd.read_csv(path_dataset)
    
    ## 2. Pseudo-Labeling
    # a. Same Pairs in Dataset
    if method=="same":
        pseudo_labels=[]
        
        for idx in df_pseudo.index:
            row=df_pseudo.loc[idx]
            
            # Encode Sentence
            enc0=tokenizer.encode(row["sent0"])
            enc1=tokenizer.encode(row["sent1"])
            
            # Forward
            if type=="bi2cross":
                pred=model(
                    torch.tensor([enc0]).to(model.pretrained.device),
                    torch.tensor([enc1]).to(model.pretrained.device)
                )
            elif type=="cross2bi":
                _input=torch.tensor([
                    enc0[:-1]+[tokenizer.sep_token_id]+enc1[1:]
                ])
                pred=model(_input.to(model.pretrained.device))
                
            pseudo_labels.append(pred[0].item())
    torch.cuda.empty_cache()
        
    # Append Pseudo-Label Column
    df_pseudo["pseudo_label"]=pseudo_labels

    ## 3. Save Dataset
    if type=="bi2cross":
        df_pseudo.to_csv("../dataset/pseudo-labels_bi2cross_loop"+str(n_loop)+".csv")
    elif type=="cross2bi":
        df_pseudo.to_csv("../dataset/pseudo-labels_cross2bi_loop"+str(n_loop)+".csv")

def distill(
    direction, # bi2cross or cross2bi
    n_loop,
    path_dataset,
    path_bi_model=None,
    path_cross_model=None,
    base_lm="roberta-base",
    device_name="cpu"
):
    """
    Distillation between Bi/Cross-Encoder
    """
    assert direction in ["bi2cross", "cross2bi"], "Wrong Direction"
    # Device
    device=torch.device(device_name)

    ## 1. Load Pre-Trained (Distilled) LM
    if direction=="bi2cross":
        assert path_bi_model!=None, "Need Pre-Trained Bi-Encoder"
    elif direction=="cross2bi":
        assert path_cross_model!=None, "Need Pre-Trained Cross-Encoder"

    # Load Bi-Encoder
    if path_bi_model==None: path_bi_model=base_lm

    tokenizer_bi=AutoTokenizer.from_pretrained(path_bi_model)
    model_bi=AutoModel.from_pretrained(path_bi_model)

    enc_bi=BiEncoder(pretrained=model_bi).to(device)
    enc_bi.eval()

    # Load Cross-Encoder
    if path_cross_model==None: path_cross_model=base_lm

    tokenizer_cross=AutoTokenizer.from_pretrained(path_cross_model)
    model_cross=AutoModel.from_pretrained(path_cross_model)

    enc_cross=CrossEncoder(pretrained=model_cross).to(device)
    enc_cross.eval()
        
    ## 2. Evaluate Loaded LM
    print(str(n_loop-1)+"th Bi-Encoder\n-----")
    # Bi-Encoder
    with torch.no_grad():
        evaluate(
            type="bi",
            tokenizer=tokenizer_bi,
            model=enc_bi
        )
    print("=====")

    if direction=="bi2cross":
        print(str(n_loop-1)+"th Cross-Encoder\n-----")
    elif direction=="cross2bi":
        print(str(n_loop)+"th Cross-Encoder\n-----")
    # Cross-Encoder
    with torch.no_grad():
        evaluate(
            type="cross",
            tokenizer=tokenizer_cross,
            model=enc_cross
        )
        
    ## 3. Pseudo-Labeling using Loaded LM
    print("\nPseudo-Labeling..")
    with torch.no_grad():
        if direction=="bi2cross":
            pseudo_label(
                type=direction,
                tokenizer=tokenizer_bi,
                model=enc_bi,
                n_loop=n_loop
            )
        elif direction=="cross2bi":
            pseudo_label(
                type=direction,
                tokenizer=tokenizer_cross,
                model=enc_cross,
                n_loop=n_loop
            )
    print("Done!")

    # 4. Distillation: Train Another (Bi <--> Cross) LM
    # 5. Evaluate Newly Distilled LM
