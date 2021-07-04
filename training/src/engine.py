import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Engine:
    def __init__(self):
        pass

    def loss_fn(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 6))

    def accuracy_threshold(self, y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
        if sigmoid:
            y_pred = y_pred.sigmoid()
        return ((y_pred > thresh) == y_true.byte()).float().mean().item()

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        # torch.cuda.manual_seed_all(seed_value)

    def train_fn(self, data_loader, model, optimizer, device, schedular):
        print("Start training...\n")
        model.train()
        batch_counts = 0
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = self.loss_fn(logits, b_labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedular.step()

    def eval_fn(self, data_loader, model, device):
        print("Starting evaluation...\n")
        model.eval()
        val_accuracy = []
        val_loss = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
                logits = model(b_input_ids, b_attn_mask)
                loss = self.loss_fn(logits, b_labels.float())
                val_loss.append(loss.item())
                accuracy = self.accuracy_threshold(logits.view(-1, 6), b_labels.view(-1, 6))
                val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy
