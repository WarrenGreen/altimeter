import pathlib

import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import RandomSampler
from tqdm import tqdm

from altimeter.dataset import JSONLDataset, SequenceDataset, split_on_key, SingleKeyDataset, NormalizeBatchDataset, \
    TensorDataset


class FCNNModel(nn.Module):
    """
    Fully-connected neural network to predict motion model.

    Args:
            k (int): sequence length considered for prediction
            dropout_prob (float):
    """
    def __init__(self, k=4, dropout_prob=0.5):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(self.k, 256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, prev_values):
        if len(prev_values) < self.k:
            raise ValueError(f"Size of input be at least `k`: {len(prev_values)} != {self.k}")
        prev_values = prev_values[:self.k]
        x = self.fc1(prev_values)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def predict(self, prev_values):
        return self.forward(prev_values)


def eval_dataset(model, dataset, device, batch_size, loss_fn, pin_memory):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    epoch_loss, epoch_acc = 0.0, 0.0
    for batch, labels in tqdm(dataloader):
        batch, labels = batch.to(device), labels.to(device)
        logits = model(batch)
        epoch_loss += loss_fn(logits, labels)

    epoch_loss /= len(dataset)
    return epoch_loss


def train(batch_size=32, epochs=20, pin_memory=True, test_frequency=5, sequence_length=4, dropout_prob=0.0):
    model = FCNNModel(k=sequence_length, dropout_prob=dropout_prob)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_filepath = str(pathlib.Path(__file__).absolute().parent.parent.parent / "data" / "poly_flight.jsonl")
    dataset = JSONLDataset(data_filepath)
    datasets = split_on_key(dataset, "split_type")
    train_dataset = SingleKeyDataset(datasets["train"], key="altitude")
    train_dataset = SequenceDataset(train_dataset, sequence_length=sequence_length)
    train_dataset = NormalizeBatchDataset(train_dataset)
    train_dataset = TensorDataset(train_dataset)
    val_dataset = SingleKeyDataset(datasets["val"], key="altitude")
    val_dataset = SequenceDataset(val_dataset, sequence_length=sequence_length)
    val_dataset = NormalizeBatchDataset(val_dataset)
    val_dataset = TensorDataset(val_dataset)
    test_dataset = SingleKeyDataset(datasets["test"], key="altitude")
    test_dataset = SequenceDataset(test_dataset, sequence_length=sequence_length)
    test_dataset = NormalizeBatchDataset(test_dataset)
    test_dataset = TensorDataset(test_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    loss_fn = MSELoss()
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch, labels in tqdm(train_dataloader):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch}: train loss: {epoch_loss}")

        val_loss = eval_dataset(model=model, dataset=val_dataset, device=device, batch_size=batch_size, loss_fn=loss_fn, pin_memory=pin_memory)
        print(f"Epoch {epoch}: validation loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f="best_model_woo!.pt")

        if epoch == 1 or epoch % test_frequency == 0:
            test_loss = eval_dataset(model=model, dataset=test_dataset, device=device, batch_size=batch_size, loss_fn=loss_fn, pin_memory=pin_memory)
            print(f"Epoch {epoch}: test loss: {test_loss}")

    torch.save(model, f="final_model.pt")


if __name__ == '__main__':
    train()
