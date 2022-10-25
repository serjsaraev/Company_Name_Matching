import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import json

class NamesDataset:
    
    def __init__(self, names_dict_path: str, tokenizer, max_length: int=512, name_example=None):
        
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.name_example = name_example
        
        if name_example is None:
        
            with open(names_dict_path, 'r') as f:
                names_dict = json.load(f)

            self.names = []
            self.labels = []
            for number, names_list in names_dict.items():
                self.names.extend([name for name in names_list])
                self.labels.extend([int(number) for _ in names_list])

            self.labels2target = {label: idx for idx, label in enumerate(set(self.labels))}
            self.labels2target['other'] = len(set(self.labels))
        else:
            self.names = [name_example, name_example]
        
    def __len__(self):
        return len(self.names)
    
    def get_name(self, idx):
        return self.names[idx]
    
    def __getitem__(self, idx):
        text = self.names[idx]
        tokenized_text = self.tokenizer.encode_plus(text, padding='max_length', truncation=True,
                                                    return_tensors="pt", max_length=self.max_length,
                                                    add_special_tokens=True)
        
        if self.name_example is None:
            label = self.labels2target[self.labels[idx]]
            return tokenized_text, label
           
        else:
            return tokenized_text, 0