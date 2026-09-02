import numpy as np
data = np.load('data/landmarks.npz')

def normalize(sample):
    pts = sample.reshape(21, 3)
    pts = pts[:, :2]
    pts = pts - pts[0]

    palm_length = np.linalg.norm(pts[0] - pts[9])
    pts = pts / palm_length

    pts = pts.flatten()
    return pts

normalized = []
for sample in data['vectors']:
    normalized.append(normalize(sample))
normalized_arr = np.array(normalized)

bad = np.abs(normalized_arr).max(axis=1) > 10
clean = normalized_arr[~bad]
clean_labels = data['labels'][~bad]



print(np.unique(data['labels'][bad], return_counts=True))
print(clean.shape, clean_labels.shape)
print(np.isnan(clean).any())
print(np.abs(clean).max())

np.savez("data/landmarks_normalized.npz", vectors=clean, labels=clean_labels)