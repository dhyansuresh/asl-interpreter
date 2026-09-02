import numpy as np
import json
from sklearn.model_selection import train_test_split

def train():
    data = np.load('data/landmarks_normalized.npz')

    keep = data['labels'] != 'nothing'

    final = data['vectors'][keep]
    final_labels = data['labels'][keep]

    classes, encoded = np.unique(final_labels, return_inverse=True)

    print(encoded.min(), encoded.max())  # 0, 27
    print(final_labels[0], encoded[0])  # 'A' and 0
    print(classes[encoded[0]])  # 'A' — back where you started

    json.dump(classes.tolist(), open('data/classes.json', 'w'))

    X_train, X_test, y_train, y_test = train_test_split(
        final, encoded, test_size=0.2, stratify=encoded, random_state=42
    )


train()