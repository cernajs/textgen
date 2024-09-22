import torch
from torch.utils.data import Dataset, DataLoader, random_split

class StoryDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()

        #split the data into entries
        entries = data.split('<|title|>')[1:]
        for entry in entries:
            try:
                title_part, story_part = entry.strip().split('<|story|>')
                title = title_part.strip()
                story = story_part.strip()
                combined_text = f"TITLE: {title}\nSTORY: {story}"
                self.examples.append(combined_text)
            except ValueError:
                continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }
