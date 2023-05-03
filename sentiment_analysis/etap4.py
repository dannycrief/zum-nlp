import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Load preprocessed data
df = pd.read_csv('../csv_files/02_preprocessed_data/preprocessed_data.tsv', sep='\t')
df['cleaned_title'] = df['cleaned_title'].astype(str)

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in df['cleaned_title']]
max_len = max([len(txt) for txt in tokenized_texts])
input_ids = torch.tensor(
    [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=max_len, truncation=True) for text
     in df['cleaned_title']])

# Create attention masks
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
attention_masks = torch.tensor(attention_masks)

# Create data loaders
batch_size = 32
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(df['cluster'].values))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_attentions=False,
                                                      output_hidden_states=False)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    for batch in train_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

# Evaluate the fine-tuned model on the test set
test_df = pd.read_csv('../csv_files/01_reddit_posts/reddit_posts_combined.tsv', sep='\t')
test_df['cleaned_title'] = test_df['cleaned_title'].astype(str)
test_inputs = torch.tensor(
    [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=max_len, truncation=True) for text
     in test_df['cleaned_title']])
test_masks = []
for seq in test_inputs:
    seq_mask = [float(i > 0) for i in seq]
    test_masks.append(seq_mask)
test_masks = torch.tensor(test_masks)
test_labels = torch.tensor(test_df['cluster'].values)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Test the model on the test set
predictions, true_labels = [], []
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)
predicted_classes = np.argmax(np.concatenate(predictions, axis=0), axis=1)
true_classes = np.concatenate(true_labels, axis=0)
accuracy = (predicted_classes == true_classes).mean()
print(f'Test Accuracy: {accuracy:.3f}')
