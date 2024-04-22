# Word Categorisation task with prominent thematic categories

from gensim.models import Word2Vec, FastText
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Define categories
categories = {
    'family_related_terms': ['pater', 'frater', 'filius', 'parens', 'mater', 'familia'],
    'religious_terms': ['deus', 'fides', 'ecclesia', 'sanctus', 'spiritus', 'christus', 'caelum', 'numen', 'sacer', 'christianus', 'pietas', 'pontifex', 'pius', 'salus', 'martyr', 'sacrus', 'hera', 'diuinus', 'templum', 'moro'],
    'royalty_terms': ['rex', 'caesar', 'princeps', 'regnum', 'regina', 'regius', 'regalis'],
    'war_related_terms': ['bellum', 'hostis', 'arma', 'miles', 'legatus', 'tribunus', 'dux', 'tyrannus'],
    'emotional_states': ['amor', 'dolor', 'gaudium', 'furia', 'metus', 'timor', 'ira', 'laetus', 'dolus', 'odium', 'spes', 'pudor', 'terror', 'rabies'],
    'body_parts_terms': ['manus', 'caput', 'pectus', 'cor', 'oculus', 'os', 'corpus', 'sanguis', 'auris', 'lingua', 'vultus'],
    'historical_terms': ['senatus', 'consul', 'praetor', 'tribunus', 'civitas', 'historia', 'princeps', 'imperium'],
    'biblical_characters': ['adam', 'eve', 'noah', 'abraham', 'moses', 'david', 'solomon', 'isaiah', 'jeremiah', 'jesus', 'peter', 'paul']
}

models_dir = "/mnt/c/TopicModelling/word embedding/third round lemma/"

def load_models_correctly(models_dir, min_size_kb=992):
    loaded_models = {}
    for filename in os.listdir(models_dir):
        # Exclude auxiliary files explicitly
        if filename.endswith(".npy") or filename.endswith(".kv"):
            continue

        filepath = os.path.join(models_dir, filename)
        if os.path.getsize(filepath) < min_size_kb * 1024:
            print(f"Skipping {filename} due to insufficient file size.")
            continue

        try:
            if "fasttext" in filename:
                model = FastText.load(filepath).wv
            else:
                model = Word2Vec.load(filepath).wv
            loaded_models[filename] = model
            print(f"Successfully loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    return loaded_models

models = load_models_correctly(models_dir)

# Iterate over each model to calculate centroids and predict categories
for model_name, model in models.items():
    centroids = {}
    for category, words in categories.items():
        vectors = [model[word] for word in words if word in model.key_to_index]
        if vectors:
            centroid = np.mean(vectors, axis=0)
            centroids[category] = centroid

    predictions = {}
    true_labels_dict = {}
    for category, words in categories.items():
        for word in words:
            if word in model.key_to_index and centroids.get(category) is not None:
                similarities = {cat: cosine_similarity([model[word]], [centroid])[0][0] for cat, centroid in centroids.items()}
                predicted_category = max(similarities, key=similarities.get)
                predictions[word] = predicted_category
                true_labels_dict[word] = category

    # Calculate metrics (optional, as it requires true labels for each word)
    true_labels = [true_labels_dict[word] for word in true_labels_dict if word in predictions]
    predicted_labels = [predictions[word] for word in true_labels_dict if word in predictions]

    precision = precision_score(true_labels, predicted_labels, labels=list(categories.keys()), average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, labels=list(categories.keys()), average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, labels=list(categories.keys()), average='macro', zero_division=0)

    print(f'Model: {model_name}')
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)