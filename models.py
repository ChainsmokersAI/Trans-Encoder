import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    """
    Bi-Encoder Model
    """
    def __init__(self, pretrained):
        super().__init__()
        
        # Pre-Trained LM
        self.pretrained=pretrained
        # Cosine Similarity
        self.cos_sim=nn.CosineSimilarity()
        
    def forward(self, x1, x2):
        # Forward Sentences "Individually"
        x1=self.pretrained(x1)
        hidden1=x1.last_hidden_state
        
        x2=self.pretrained(x2)
        hidden2=x2.last_hidden_state
        
        # Compute Similarity
        cos_sims=self.cos_sim(hidden1[:,0,:], hidden2[:,0,:]).unsqueeze(-1)
        return cos_sims

class CrossEncoder(nn.Module):
    """
    Cross-Encoder Model
    """
    def __init__(self, pretrained):
        super().__init__()
        
        # Pre-Trained LM
        self.pretrained=pretrained
        # Pooling Layer: Compute Similarity
        self.pooler=nn.Linear(pretrained.config.hidden_size, 1)
        
    def forward(self, x):
        # Forward Concatenated Sentences "at Once"
        x=self.pretrained(x)
        cls=x.last_hidden_state[:,0,:]

        # Return Similarity
        return self.pooler(cls)
