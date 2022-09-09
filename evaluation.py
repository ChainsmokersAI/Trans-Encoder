import torch

import numpy as np
from scipy import stats

def evaluate(type, tokenizer, model, dataset_path="../dataset/stsbenchmark/sts-test.csv"):
    """
    Evaluate Pre-Trained (Distilled) LM on STS Task
    """
    # Dataset: STS Benchmark Test Set
    with open(dataset_path, "r") as f:
        test_set=f.read()
        f.close()

    preds=[]
    labels=[]
    for data in test_set.split("\n")[:-1]:
        label, sent0, sent1=data.split("\t")[4:7]
        labels.append(float(label))

        # Encode Sentence
        enc0=tokenizer.encode(sent0)
        enc1=tokenizer.encode(sent1)

        # Forward
        if type=="bi":
            pred=model(
                torch.tensor([enc0]).to(model.pretrained.device),
                torch.tensor([enc1]).to(model.pretrained.device)
            )
        elif type=="cross":
            _input=torch.tensor([
                enc0[:-1]+[tokenizer.sep_token_id]+enc1[1:]
            ])
            pred=model(_input.to(model.pretrained.device))
            
        preds.append(pred[0].item())
    torch.cuda.empty_cache()
        
    print(np.corrcoef(preds, labels))
    print(stats.spearmanr(preds, labels))
