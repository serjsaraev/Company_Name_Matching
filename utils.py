import torch
from torch.utils.data import DataLoader

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import faiss

from tqdm import tqdm

from dataset import NamesDataset




def fit_epoch(model, data_loader, criterion, optimizer, device, scheduller):
    
    model.train()
    train_loss = []
    for idx, (texts, labels) in tqdm(enumerate(data_loader)):
        texts = {k: v.to(device) for k, v in texts.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        embs, output = model(
            torch.squeeze(texts['input_ids']), torch.squeeze(texts['attention_mask']), labels)
        
        loss = criterion(output, labels)
        loss.backward()
        
        optimizer.step()
        
        scheduller.step()
        
        train_loss.append(loss.item())
        
        if idx % 20 == 0:
            print('Train loss:', np.mean(train_loss))
            
        
    return np.mean(train_loss)

@torch.no_grad()
def make_embeddings(model, data_loader, device, emb_size):
    
    model.eval()
    embeddings = np.zeros(shape=(len(data_loader.dataset), emb_size))
    y_trues = []
    
    idx = 0
    for texts, labels in tqdm(data_loader):
        texts = {k: v.to(device) for k, v in texts.items()}

        embs = model(
            torch.squeeze(texts['input_ids']), torch.squeeze(texts['attention_mask']))
        
        y_trues.extend([label for label in labels])
        for emb in embs:
            embeddings[idx] = emb.to('cpu').numpy()
            idx += 1
    
    embeddings = np.float32(embeddings)
    embeddings = normalize(embeddings, axis=1, norm='l2')

    return embeddings, y_trues

def predict(embeddings: np.ndarray, centroids: np.ndarray, topk: int):

    similarities = cosine_similarity(embeddings, centroids)
    torch_similarities = torch.from_numpy(similarities).float()
    confs, preds = torch.topk(torch_similarities, topk, dim=1)
    confs, preds = confs.numpy(), preds.numpy()

    return confs, preds


def compute_all_val_metrics(model, embeddings: np.ndarray, centroids: np.ndarray, classes: np.ndarray,
                                y_val: list, other_label):

        ## Получаем предсказания модели
        confs, preds = predict(embeddings=embeddings, centroids=centroids, topk=3)

        ## Преобразуем полученные предсказания в предсказанные метки классов
        preds = [[classes[p] for p in pred] for pred in preds]

        best_acc_3 = 0
        for tresh in [0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8]:  ## Подбираем лучший порог для отсечения альтернативного класса
            new_preds = []
            for idx, (pred, conf, true) in enumerate(zip(preds, confs, y_val)):
                y_pred = []
                for p, c in zip(pred, conf): ## Проверяем каждое предсказание
                    if c >= tresh:
                        y_pred.append(p) ## если сходство больше порога отсавляем предсказанную метку
                    else:
                        y_pred.append(other_label) ## иначе, заменяем предсказание на метку альтернативного класса
                if true in y_pred: ## Проверяем есть ли правильная метка в топ 3 предсказаниях модели
                    new_preds.append(true)
                else:
                    new_preds.append(y_pred[0])

            new_preds = np.array(new_preds)
            acc = accuracy_score(y_val, new_preds) ## Рассчитываем долю правильных ответов

            if acc > best_acc_3:
                best_acc_3 = acc ## Выбираем метрику для наилучшего порога

        return best_acc_3



@torch.no_grad()
def eval_epoch(model, train_loader, val_loader, device, emb_size, other_label):
    
    train_embeddings, y_train = make_embeddings(model, train_loader, device, emb_size)
    val_embeddings, y_val = make_embeddings(model, val_loader, device, emb_size)
    
    clf = NearestCentroid(metric='cosine') 
    clf.fit(train_embeddings, y_train)
    
    best_acc_3 = compute_all_val_metrics(model=model, embeddings=val_embeddings, centroids=clf.centroids_,
                                         classes=clf.classes_, y_val=y_val, other_label=other_label)
    
    
    return best_acc_3


@torch.no_grad()
def get_faiss_index(model, data_loader, device, emb_size):
    
    embeddings, y_trues = make_embeddings(model, data_loader, device, emb_size)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return index

def get_similarity_names(name, model, index, sim_names_cnt, tokenizer, device, emb_size, thresh):
    
    name_dataset = NamesDataset('', tokenizer=tokenizer, max_length=32, name_example=name)
    data_loader = DataLoader(name_dataset, batch_size=2, shuffle=False)
    val_embeddings, y_val = make_embeddings(model, data_loader, device, emb_size)
    
    D, I = index.search(val_embeddings, k=sim_names_cnt)
    
    result = []
    for dist, idx in zip(D[0], I[0]):
        if dist >= thresh:
            result.append((idx, dist))
    
    return result





    