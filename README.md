### **ESG NLP Classification – In-House Competition (Signate)**

**Task:** Multi-class classification of ESG (Environmental, Social, Governance) categories in corporate 10-K reports.
**Evaluation Metric:** F1 Score

#### Exploratory Data Analysis (EDA)

* Gained domain understanding of ESG-related topics.
* Analyzed dataset for class imbalance, missing values, and noise.
* Visualized feature distributions, label frequencies, and word usage patterns using word clouds and histograms.

#### Data Challenges and Characteristics

* Highly imbalanced dataset with noisy labels and incomplete context.
* Sentences were extracted independently from documents, requiring contextual aggregation.
* Ensured data integrity by applying GroupKFold to avoid document leakage between train and validation splits.

#### Data Preparation

* Cleaned and preprocessed text.
* Performed undersampling by merging repetitive sequences and oversampling by creating valid sentence combinations using metadata (paragraph IDs, titles, labels).
* Generated pseudo-labels for minority classes based on high-confidence predictions.
* Added custom tokens to represent document metadata and section types.

#### Model Training

* Fine-tuned ALBERT, DeBERTa, and RoBERTa models using Hugging Face Transformers.
* Applied:

  * Layer freezing and layer-wise learning rate schedules
  * Custom separators for embedding document metadata
  * Stochastic Weight Averaging (SWA)
  * Pooling strategies using the last four hidden layers (average, max, concat)

#### Ensemble Method

* Combined high-performing models using weighted averaging to improve robustness and generalization.

#### Technical Challenges

* Encountered overfitting due to initial data leakage.
* Faced memory constraints on GPU during training; addressed using:

  * Gradient accumulation
  * Mixed precision training
  * Smaller batch sizes and models
* Experienced difficulty reproducing results due to inconsistencies in SWA checkpoints.
* Lack of a modular training pipeline and proper experiment tracking.

#### Lessons Learned

* Reinforced the importance of reproducible ML workflows, modular pipeline design, and structured logging.
* Identified the value of frameworks like Weights & Biases and PyTorch Lightning for efficient model development and monitoring.

