import torch
import math
from torch.nn import functional as F
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AutoTokenizer, AutoModel

from torch.nn import Parameter

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


class ArcFace(nn.Module):
    """ Класс для определения слоя ArcFace """
    def __init__(self, in_features, out_features, scale_factor=64.0, margin=0.50, criterion=None):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        return logit



class NamesRecognition(nn.Module):
    
    def __init__(self, feature_extractor_name: str, embedding_size: int, margin: float, scale: int,
                num_of_classes: int, loss_func: str='arcface'):
        super(NamesRecognition, self).__init__()
        AdaCos
        self.feature_extractor = BertModel.from_pretrained(feature_extractor_name)
        self.head = nn.Linear(768*32, embedding_size)
        
        if loss_func == 'arcface':
            self.arc = ArcFace(
                in_features=embedding_size,
                out_features=num_of_classes,
                scale_factor=scale,
                margin=margin
            )
        elif loss_func == 'adacos':
            self.arc = AdaCos(
                num_features=embedding_size,
                num_classes=num_of_classes,
                m=margin
            )
        
        self.flatten = nn.Flatten()
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        
        features = self.feature_extractor(input_ids, attention_mask).last_hidden_state
        features = self.flatten(features)
        emb = self.head(features)
        
        
        if labels is not None:
            output = self.arc(emb, labels)
            return emb, output
        
        else:
            return emb