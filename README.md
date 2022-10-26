## Company Names Matching
Задача: Поиск похожих названий компаний из базы данных.

### 1. Данные
Для решения задачи использовался [датасет](data/train.csv) с парами наименований компаний.
### 2. Подходы решения
#### 2.1. Bert + ArcFace
Подход заключается в обучении модели классификации наименований компаний. Для этого необходимо из исходного датасета получить новый датасет, который будет содержать в себе классы, каждый из которых содержит в себе различные варианты наименований определенной компании. Получили [новый датасет](data/names_dict.json).

В качестве Feature Extractor используем Bert. Затем добавляем модуль ArcFace и обучаем модель классификации. На инференсе осталяем только Bert FeatureExtractor для получения эмбеддиногов для наименований.
Для оценки качества использовали метрику Accuracy top3.

| Model       | ACC@3   |
|:------------|:--------|
| BertArcFace | 93.0 %  |
| BertAdaCos  | 91.6  % | 

Код обучения: [train_notebook.ipynb](train_notebook.ipynb).

Веса моделей: [https://disk.yandex.ru/d/gMLzPNPF3GnfaQ](https://disk.yandex.ru/d/gMLzPNPF3GnfaQ)
#### 2.2. Sentence-transformer

В данном подходе попробовали fine-tune модели [distilbert-base-nli-mean-tokens](https://huggingface.co/sentence-transformers/distilbert-base-nli-mean-tokens).

После дообучения в 10 эпох (больше не позволил colab) получили результаты: [binary_classification_evaluation_results.csv](binary_classification_evaluation_results.csv)

Код обучения: [sentence_transformers.ipynb](sentence_transformers.ipynb).

Веса моделей: [Google Drive](https://drive.google.com/drive/folders/1-oRgvgJUiggIQodD5NvB_2u84ulzTTgm?usp=sharing)

### 3. Инференс
Создаем базу эмбеддиногов полученных из наименований компаний за счет Bert FeatureExtractor. Для любого нового наименования применяем faiss для поиска 5 наиболее похожих наименований из базы.

Код инференса: [inference_notebook.ipynb](inference_notebook.ipynb)
