import numpy as np
import json

import torch
from sklearn.model_selection import train_test_split
from torch import nn

def prepare_data():
    data = np.load('data/landmarks_normalized.npz')

    keep = data['labels'] != 'nothing'

    final = data['vectors'][keep]
    final_labels = data['labels'][keep]

    classes, encoded = np.unique(final_labels, return_inverse=True)

    json.dump(classes.tolist(), open('data/classes.json', 'w'))

    X_train, X_test, y_train, y_test = train_test_split(
        final, encoded, test_size=0.2, stratify=encoded, random_state=42
    )

    return X_train, X_test, y_train, y_test, classes


def train(X_train, X_test, y_train, y_test):
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    model = nn.Sequential(
        nn.Linear(42, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 28),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    out = model(X_train[:5])
    print(out)

X_train, X_test, y_train, y_test, classes = prepare_data()
train(X_train, X_test, y_train, y_test)