import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Function to batch and average data
def batch_average(data, batch_size):
    return [np.mean(data[i:i + batch_size]) for i in range(0, len(data), batch_size)]

# Coherence scores for MALLET
mallet_coherence = []

# Coherence scores for Top2Vec
top2vec_coherence = []

# Coherence scores for Scikit-lda
scikit_lda_coherence = []

plt.figure(figsize=(14, 8))

# Batching size - adjust as needed
batch_size_mallet = len(mallet_coherence) // len(top2vec_coherence)
batch_size_scikit = len(scikit_lda_coherence) // len(top2vec_coherence)

# Batch and average
mallet_batched = batch_average(mallet_coherence, batch_size_mallet)
scikit_batched = batch_average(scikit_lda_coherence, batch_size_scikit)

# Create index for batched data
x_mallet = np.linspace(0, len(top2vec_coherence) - 1, len(mallet_batched))
x_scikit = np.linspace(0, len(top2vec_coherence) - 1, len(scikit_batched))

# Plot batched coherence scores
plt.plot(x_mallet, mallet_batched, label='MALLET', marker='o')
plt.plot(range(len(top2vec_coherence)), top2vec_coherence, label='Top2Vec', marker='o')
plt.plot(x_scikit, scikit_batched, label='Scikit-lda', marker='o')

plt.xlabel('Normalized Configuration Index')
plt.ylabel('Coherence Score')
plt.title('Normalized Coherence Scores Across Different Topic Modeling Algorithms')
plt.legend()
plt.grid(True)
plt.show()