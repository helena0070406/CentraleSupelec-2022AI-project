from typing import List, Dict, Tuple

import torch

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

class AspectTermDataset(Dataset):
    def __init__(self, file_name, tokenizer):
        # Load and process data
        self.data = []
        self.tokenizer = tokenizer
        with open(file_name, 'r') as file:
            for line in file:
                polarity, aspect_category, term, _, sentence = line.strip().split('\t')
                inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
                label = {'positive': 0, 'negative': 1, 'neutral': 2}[polarity]
                self.data.append((inputs, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Get the item
        item = self.data[idx]

        # Get the inputs and label
        inputs, label = item

        # Convert input format
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}, label

    
    def pad_collate(batch):
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []

        # Find the maximum input length in the batch
        max_input_len = max([item[0]['input_ids'].size(0) for item in batch])

        # Pad the input_ids and attention_mask tensors
        for item in batch:
            inputs = item[0]
            label = item[1]

            # Pad input_ids
            padding_len = max_input_len - inputs['input_ids'].size(0)
            padded_input_ids = torch.cat([inputs['input_ids'], torch.zeros(padding_len, dtype=torch.long)], dim=0)
            batch_input_ids.append(padded_input_ids)

            # Pad attention_mask
            padded_attention_mask = torch.cat([inputs['attention_mask'], torch.zeros(padding_len, dtype=torch.long)], dim=0)
            batch_attention_mask.append(padded_attention_mask)

            # Add label
            batch_labels.append(label)

        # Stack the tensors
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return batch_input_ids, batch_attention_mask, batch_labels

    

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 3)

    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        # Load data
        train_dataset = AspectTermDataset(train_filename, self.tokenizer)
        dev_dataset = AspectTermDataset(dev_filename, self.tokenizer)

        # Create DataLoaders
        #train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=AspectTermDataset.pad_collate)
        dev_dataloader = DataLoader(dev_dataset, batch_size=16)

        # Move model to device
        self.model.to(device)

        # Set up optimizer and loss function
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        loss_function = CrossEntropyLoss()

        # Train the model
        self.model.train()
        num_epochs = 1 # only 1 epoch per run in order to minimize exec time
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Calculate loss and gradients
                loss = outputs.loss
                loss.backward()

                # Update model parameters
                optimizer.step()


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Load data
        test_dataset = AspectTermDataset(data_filename, self.tokenizer)

        # Create DataLoader
        batch_size = 16
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=AspectTermDataset.pad_collate)

        # Move model to device
        self.model.to(device)

        # Set model to eval mode
        self.model.eval()

        # Store predictions
        predictions = []

        # Predict labels
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get predicted labels
                predicted_labels = torch.argmax(outputs.logits, dim=-1)

                # Convert label indices to labels
                label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
                predicted_labels = [label_map[label.item()] for label in predicted_labels]

                # Append predictions to the list
                predictions.extend(predicted_labels)

        return predictions