### ESG_NLPClassification
This is some of the approaches and code for inhouse competition of "NLP competition for Green" .
(dataset and informaiton about the data are not uploaded.)


Task :
NLP  multi-class classification  for classifying content for 10Kdata set.

Metrics : F1 score

Example data set format.

| id | document | sentencce | label
| --- | ----------- | -------- | -----
| 1 | doc1 | sentence1 | 1
| 2 | doc2 | sentence2 | 0
| 3 | doc3 | sentence3 | 1
| 4 | doc4 | sentence4 | 3


#### EDA

Try to understand the domain(about ESG) and task.
Analyze the dataset and features.
 - Detect data type, outliers, etc
 - Identity missing data
 - Visualize distribution of data as a whole, for each features , for each label ,etc.
 - Use wordcloud to visualize the most common words for each category


#### Train-Validation Split

 Each record in the dataset is a sentence from the 10k document.The records of the the same document shoudn't appear in different fold.
 1. Try GroupKFold
 2. After analysing and visualization the class distribution for each document,manually select document for training and validation. 


#### Data 

 Very unbalanced dataset. Noises in data.Cannot make prediction just on signle records. 

Approach: 
- Clean text data
- Used oversampling , undersampling to solve imbalance.
- Generate new features such as title,paragraph group from the original html file .
- Create different record combination based on new information.
- Remap newly created text data with original id and label from train.tsv
- Adding new special token


Undersampling (majority class)
- Ramdomly select subsect of data instead of all.
- Merge multiple consecutive same class records into single record.
	
Oversampling (mainority classes)
- Generate new sentences to increase small label data size.
- Different combination of text based on paragraph ids, title, label etc.Skip if the text combination does't make sense.
- Create Psuedolabel for small label data.(Train -> Predict testdata set -> use data with high confidence as a pseudolabel)

#### Model Training
  FineTune Albert,Deberta,Roberta Model, Futher pretraining Roberta on the train dataset than finetune.

##### Roberta
- Freezing different layers
- Layerwise learnig rate trainig
- Adding metadata information with new seperator and with category information
- Stochastic Weight Averaging
- Using Last 4 layers (average pooling, max pooling, concat)
- Frequent model evaluation 
 
  
BERT-based Ensembles
##### Essembly
- Weighted averge of multiple high accuracy model.


#### Challenges
- Didin't have pipeline -problem with managing code and tracking score

- Overfitting (Data Leakage problem in model of the final submission , highest private leaderboard - 23th )

- Problems with saving, swa pytorch model to reproduce the code or resume traininig

- Longer sequence

- Finding learning rates 


- RuntimeError: CUDA error: out of memory
  
  Solution for small memory :
	- Use different batch size for local GPU and cloud GPU.
	- Gradient Accumulation
	- Automatic Mixed Precision
	- Reduce batched size
	- Use smaller pretrained model and data
	
####  Take away,

Prepare pipeline and common script,visualization, logging or use framework for Machine learning model management such as WandB, PytorchLighting,etc





###  References

##### Model Fine Tuning


https://ruder.io/recent-advances-lm-fine-tuning/

https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/A-Primer-in-BERTology-What-We-Know-About-How-BERT

https://arxiv.org/abs/1905.05583 (How to Fine-Tune BERT for Text Classification?)

https://linuxtut.com/en/343309257da1798c1b63/ (10 methods to improve the accuracy of BERT)

https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb


##### GPU
https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e


##### Other
https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

https://www.youtube.com/watch?v=-ix_Mjzu8BU&t=180s (Ensemble of networks for improved accuracy in deep learning)

https://www.kaggle.com/rhtsingh/code (Roberta finetuning)


#### Learned resources


https://forums.fast.ai/t/adding-structured-metadata-to-text-classification-models/42906 


https://arxiv.org/pdf/1909.08402v1.pdf (Enriching BERT with Knowledge Graph Embeddings for Document)

https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z  (Stanford CS224N: NLP with Deep Learning | Winter 2019 | )

https://www.youtube.com/watch?v=iDulhoQ2pro (Attention Is All You Need)

https://www.youtube.com/watch?v=-9evrZnBorM  (BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)

https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz (Pytorch)

https://www.youtube.com/channel/UCSNeZleDn9c74yQc-EKnVTA (Kaggle)
