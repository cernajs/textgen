import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import random_split

from dataset import StoryDataset

MODEL = 'gpt2-medium'

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)

#set pad token since gpt2 doesn't have one
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL)
model.resize_token_embeddings(len(tokenizer))

PATH = "blog_data_short.txt"
#PATH = "blog_data.txt"

dataset = StoryDataset(PATH, tokenizer, max_length=512)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    evaluation_strategy='steps',
    eval_steps=20,
    save_steps=500,
    logging_steps=20,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

trainer.train()

trainer.save_model('./model')
tokenizer.save_pretrained('./model')
