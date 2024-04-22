### Top2Vec ###

# Load documents

import os
import re

documents_list = []
for filename in os.listdir("/mnt/c/TopicModelling/corpus/lat/checked/chunked_full"):
    with open(os.path.join("/mnt/c/TopicModelling/corpus/lat/checked/chunked_full", filename), 'r') as f:
        text = f.read()
        documents_list.append(text)
len(documents_list)

# Training

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from top2vec import Top2Vec
import itertools

texts = [[word for word in document.lower().split()] for document in documents_list]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Define parameter space
embedding_models = ['universal-sentence-encoder', 'distiluse-base-multilingual-cased']
umap_args_list = [
    {'n_neighbors': 15, 'n_components': 5, 'metric': 'cosine'},
    {'n_neighbors': 5, 'n_components': 15, 'metric': 'euclidean'}
]
hdbscan_args_list = [
    {'min_cluster_size': 5, 'metric': 'euclidean', 'cluster_selection_epsilon': 0.0},
    {'min_cluster_size': 10, 'metric': 'euclidean', 'cluster_selection_epsilon': 0.5}
]

results = []

for embedding_model, umap_args, hdbscan_args in itertools.product(embedding_models, umap_args_list, hdbscan_args_list):
    model = Top2Vec(documents=documents_list, 
                    speed="learn", 
                    embedding_model=embedding_model, 
                    umap_args=umap_args, 
                    hdbscan_args=hdbscan_args)
    
    num_topics = model.get_num_topics()
    
    # Retrieve the topics and their words from the model
    topic_words, word_scores, topic_nums = model.get_topics()
    top_words = [words for words, scores, nums in zip(topic_words, word_scores, topic_nums)]

    # Calculate the coherence score
    coherence_model = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    
    results.append({
        'embedding_model': embedding_model,
        'umap_args': umap_args,
        'hdbscan_args': hdbscan_args,
        'num_topics': num_topics,
        'coherence_score': coherence_score
    })

for result in results:
    print(f"Model Configuration: Embedding Model={result['embedding_model']}, UMAP Args={result['umap_args']}, HDBSCAN Args={result['hdbscan_args']}")
    print(f"Number of Topics: {result['num_topics']}, Coherence Score: {result['coherence_score']}\n")

# Save results in a df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(results)
df['configuration'] = df.apply(lambda x: f"{x['embedding_model']}, {x['umap_args']['n_neighbors']} neighbors, {x['hdbscan_args']['min_cluster_size']} min cluster", axis=1)

# Visualisation
    
plt.figure(figsize=(6, 8))
plt.plot(df_sorted['configuration'], df_sorted['coherence_score'], marker='o', linestyle='-', color='skyblue')
plt.xticks(rotation=90)  # Rotate labels to avoid overlap
plt.xlabel('Configuration')
plt.ylabel('Coherence Score')
plt.title('Coherence Scores Across Different Top2Vec Configurations')
plt.ylim(0, 1)  # Ensure the y-axis ranges from 0 to 1 for coherence scores
plt.grid(True)
plt.tight_layout()
plt.show()
