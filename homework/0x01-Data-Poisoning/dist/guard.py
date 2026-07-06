import joblib
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

train = np.load(os.path.join(BASE, "train.npy"))
discriminator = joblib.load(os.path.join(BASE, "discriminator.joblib"))
THRESH = 30
NUM = 4000

def pairwise_hamming(A, B):
    """Pixel-count (Hamming) distance between every row of A and every row of B,
    i.e. L1 on binary images. One matmul: hamming(a, b) = |a| + |b| - 2*(a . b).
    A, B are binary float32 arrays of shape (N, 784) / (M, 784); returns (N, M)."""
    return A.sum(1)[:, None] + B.sum(1)[None, :] - 2.0 * (A @ B.T)

def closest_pair(D):
    """(min distance, row i, col j) of a distance matrix D."""
    i, j = np.unravel_index(D.argmin(), D.shape)
    return D[i, j], i, j

def load_data(image_path, label_path):
    # The npz should be valid
    try:
        arr = np.load(image_path)
    except:
        return False, f"Error loading images"
    try:
        labels = np.load(label_path)
    except:
        return False, f"Error loading labels"
    
    # The images should be a 2D array of shape (N, 784)
    if len(arr.shape) != 2 or arr.shape[1] != 28 * 28:
        return False, f"Expected input shape: (N, 784), got {arr.shape}"
    
    # The labels should be a 1D array of shape (N,)
    if len(labels.shape) != 1 or labels.shape[0] != arr.shape[0]:
        return False, f"Expected labels shape: ({arr.shape[0]},), got {labels.shape}"
    
    # There should be a limited number of images
    if arr.shape[0] > NUM:
        return False, f"Expected at most {NUM} images, got {arr.shape[0]}"

    # The image should be binary (0 or 1)
    if not np.all(np.isin(arr, [0, 1])):
        return False, f"Expected images to be binary (0 or 1), got {np.unique(arr)}"

    # The labels should be 0 or 1
    if not np.all(np.isin(labels, [0, 1])):
        return False, f"Expected labels to be 0 or 1, got {np.unique(labels)}"
    
    # Every submitted image must be far from every training image
    A = arr.astype(np.float32)
    T = train.astype(np.float32)
    dist, i, _ = closest_pair(pairwise_hamming(A, T))
    if dist <= THRESH:
        return False, f"Image {i} is only {int(dist)} px from a training image (need L1 > {THRESH})"
    
    # Every submitted image must be far enough from every other submitted image
    pairwise_dist = pairwise_hamming(A, A)
    np.fill_diagonal(pairwise_dist, np.inf)
    dist, i, j = closest_pair(pairwise_dist)
    if dist <= 2:
        return False, f"Images {i} and {j} are only {int(dist)} px apart (need L1 > 2)"

    # Every submitted image must be predicted as real by the discriminator
    pred = discriminator.predict(arr)
    if not np.all(pred == 1):
        bad = np.where(pred == 0)[0]
        return False, f"Image {bad[0]} is predicted as fake by the discriminator"

    print("All checks passed")
    return True, (arr, labels)