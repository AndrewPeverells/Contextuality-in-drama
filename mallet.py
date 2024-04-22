### MALLET ###

from gensim.models.wrappers import LdaMallet
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import os
from gensim import corpora
import itertools

path_to_mallet_binary = '/mnt/c/TopicModelling/mallet/bin/mallet'

texts = [[word for word in document.lower().split()] for document in documents_list]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Define parameter space
n_topics = [5, 10, 20, 30]
optimize_interval_values = [50, 100, 300, 500, 1000, None]
iterations_values = [100, 200, 300, 400, 500, 1000]

results = []

for topics, optimize_interval, iterations in itertools.product(n_topics, optimize_interval_values, iterations_values):
    # Initialize the Mallet model
    if optimize_interval is None:
        optimize_interval = 0  # Gensim's wrapper expects 0 for no optimization
    mallet_model = LdaMallet(path_to_mallet_binary, corpus=corpus, num_topics=topics, id2word=dictionary,
                             optimize_interval=optimize_interval, iterations=iterations)
    
    # Calculate coherence score
    coherence_model = CoherenceModel(model=mallet_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    results.append({
        'n_topics': topics,
        'optimize_interval': optimize_interval,
        'iterations': iterations,
        'coherence_score': coherence_score
    })

for result in results:
    print(f"Configuration: {result['n_topics']} topics, Optimize Interval: {result['optimize_interval']}, Iterations: {result['iterations']}, Coherence: {result['coherence_score']}")

# Visualisation
    
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'results' is your list of dictionaries containing the configurations and their coherence scores
df = pd.DataFrame(results)

# Creating a more descriptive label for each configuration
df['configuration'] = df.apply(lambda x: f"{x['n_topics']} topics, Optimize Interval: {x['optimize_interval'] if x['optimize_interval'] != 0 else 'None'}, Iterations: {x['iterations']}", axis=1)

# Sorting by coherence scores in ascending order
df.sort_values('coherence_score', ascending=True, inplace=True)

# Switching to a linear visualization
plt.figure(figsize=(24, 10))
plt.plot(df['configuration'], df['coherence_score'], marker='o', linestyle='-')
plt.xticks(rotation=90)  # Rotate labels to avoid overlap
plt.xlabel('Configuration')
plt.ylabel('Coherence Score')
plt.title('Coherence Scores Across Different Mallet Model Configurations')
plt.ylim(0, 1)  # Scale the y-axis from 0 to 1
plt.grid(True)
plt.tight_layout()
plt.show()