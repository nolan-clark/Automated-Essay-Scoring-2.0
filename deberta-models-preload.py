import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from datasets import Dataset, DatasetDict
import unicodedata
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# make sure GPU active
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class PATHS:
    train_path = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv'
    test_path = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv'
    model_path = "microsoft/deberta-v3-small" # pretrained model to fine-tune

# Set configurations of tokenizer and trainer
class CFG:
    max_length = 1024
    train_batch_size = 2
    eval_batch_size = 2

# Initialize Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path)
model = AutoModelForSequenceClassification.from_pretrained(PATHS.model_path, num_labels=1)

# Update dropout prob to set model classification to Regression (num_labels=1)
model.config.attention_probs_dropout_prob = 0.0
model.config.hidden_dropout_prob=0.0

# Add new tokens because DeBERTa removes "new paragraph" and "double space" from essay
tokenizer.add_tokens([AddedToken("\n", normalized=False)])
tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])

# data import
train_df = pd.read_csv(PATHS.train_path)
test_df = pd.read_csv(PATHS.test_path)

print(train_df['score'].value_counts())

def preprocess_text(text):
    # Normalize and remove unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8','ignore')
    return text.strip()

# Preprocess data: strip whitespace 
train_df['full_text'] = train_df['full_text'].apply(preprocess_text)
test_df['full_text'] = test_df['full_text'].apply(preprocess_text)

# set labels to [0,1,2,3,4,5]
train_df['score'] = train_df['score'] - 1
train_df.rename(columns={'score':'labels'},inplace=True)
train_df['labels']=train_df['labels'].astype('float32')

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Tokenize essays
def tokenize_function(examples):
    return tokenizer(examples['full_text'], padding=True, 
                                    truncation = True,
                                    max_length=CFG.max_length, 
                                    return_tensors='pt')

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/kaggle/input/deberta-models-preload/',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type = 'linear',
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy='no',
    optim = 'adamw_torch',
    report_to='none'
)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    return {
        'quadratic_weighted_kappa': cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Save fine-tuned model and tokenizer
model.save_pretrained('/kaggle/input/deberta-models-preload/model')
tokenizer.save_pretrained('/kaggle/input/deberta-models-preload/token')


# show performance metrics on train data
predictions=trainer.predict(tokenized_datasets['train']).predictions
preds=predictions.round().astype(int).squeeze()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(train_df['labels'], preds)

ConfusionMatrixDisplay(cm).plot()
plt.show()
print('QWK Score: ',cohen_kappa_score(train_df['labels'], preds, weights='quadratic'))


# --- Generate Optimized Thresholds for Classification ----

import optuna
from sklearn.model_selection import KFold

class OptunaRounder:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = np.unique(y_true)

    def __call__(self, trial):
        thresholds = []
        for i in range(len(self.labels) - 1):
            low = max(thresholds) if i > 0 else min(self.labels)
            high = max(self.labels)
            t = trial.suggest_float(f't{i}', low, high)
            thresholds.append(t)
        try:
            opt_y_pred = self.adjust(self.y_pred, thresholds)
        except: return 0
        return cohen_kappa_score(self.y_true, opt_y_pred, weights='quadratic')

    def adjust(self, y_pred, thresholds):
        opt_y_pred = pd.cut(y_pred,
                            [-np.inf] + thresholds + [np.inf],
                            labels=self.labels)
        return opt_y_pred

y = train_df['labels']
preds_valid = preds
optuna.logging.set_verbosity(optuna.logging.WARNING) 
objective = OptunaRounder(y - y.min(), preds_valid - y.min())
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, timeout=100)

best_thresholds = sorted(study.best_params.values())
print(f'Optimized thresholds: {best_thresholds}')

# Round predictions by thresholds
predictions_opt = pd.cut(preds_valid, [-np.inf] + best_thresholds + [np.inf], labels=[0,1,2,3,4,5])
preds_opt = predictions.astype(int)

cm = confusion_matrix(y, preds_opt)

ConfusionMatrixDisplay(cm).plot()
plt.show()

# Display the Optuna Threshold lines
fig=plt.hist(predictions, bins = 500)
plt.vlines(best_thresholds, ymin=0,ymax=fig[0].max(), color='k')
plt.show()
print('QWK Score: ',cohen_kappa_score(y, preds_opt, weights='quadratic'))
