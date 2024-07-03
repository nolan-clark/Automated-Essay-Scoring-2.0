# import packages
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from datasets import Dataset
import unicodedata
import gc

# display inputs to confirm correct paths
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# make sure GPU active
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# fine-tuned model paths
class PATHS:
    test_path = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv'
    linear_deberta_path = '/kaggle/input/deberta-models-preload/'
    linear_longformer_path = '/kaggle/input/longformer-models-preload/'
    linear_xlnet_path = '/kaggle/input/xlnet-models-preload/'
        

def preprocess_text(text):
    # Normalize and remove unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8','ignore')
    return text.strip()

# max embedding size configurations of each model
max_values={
    PATHS.linear_deberta_path : 1024,
    PATHS.linear_longformer_path : 2048,
    PATHS.linear_xlnet_path : 1024,
}

# import dataset for predictions
test_df = pd.read_csv(PATHS.test_path)
test_df['full_text'] = test_df['full_text'].apply(preprocess_text)
test_dataset = Dataset.from_pandas(test_df)

# model and tokenizer paths
paths = [PATHS.linear_longformer_path, PATHS.linear_deberta_path, PATHS.linear_xlnet_path]

bucket = []
for path in paths:
    # Initialize fine-tuned Model and Tokenizer from pretrained path
    model=AutoModelForSequenceClassification.from_pretrained(path+'model', num_labels=1)
    tokenizer=AutoTokenizer.from_pretrained(path+'token')

        
    def tokenize_function(examples):
        return tokenizer(examples['full_text'], 
                     padding=True,
                     #pad_to_multiple_of=512,
                     max_length= max_values[path],
                     truncation=True,
                     return_tensors='pt')
        
# Null training arguments -- only performing predictions        
    training_args = TrainingArguments(
        ".",
        per_device_eval_batch_size=1,
        report_to='none'
    )
    
# Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator= DataCollatorWithPadding(tokenizer)
        )
    ds = test_dataset.map(tokenize_function, batched=True).remove_columns(['essay_id','full_text'])
    predictions = trainer.predict(ds)
    
    bucket.append(predictions.predictions.squeeze())
    
# empty GPU each iteration
    del model
    del tokenizer
    gc.collect()

# Aggregate all predictions and classify by optimized thresholds
mean_preds=np.mean(bucket,axis=0)
best_thresholds = [0.8908, 1.7609, 2.6817, 3.5793, 4.5426] # thresholds from Optuna hypertuning
predictions = pd.cut(mean_preds.squeeze(), [-np.inf] + best_thresholds + [np.inf], labels=[0,1,2,3,4,5])
preds = predictions.astype(int)
final_predictions = preds+1

# merge predictions with essay ids for submission
sub = test_df[['essay_id']].join(pd.DataFrame({'score':final_predictions.clip(1,6)}))
sub.to_csv('submission.csv', index=False)
sub.head(5)