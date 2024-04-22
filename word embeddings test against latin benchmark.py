import os
from gensim.models import KeyedVectors

# Get all .kv files in the directory
kv_files = [f for f in os.listdir(directory) if f.endswith('.kv')]
models = {}
for kv_file in kv_files:
    model_name = os.path.splitext(kv_file)[0]
    kv_path = os.path.join(directory, kv_file)
        
    # Check file size before attempting to load
        if os.path.getsize(kv_path) > MIN_FILE_SIZE_BYTES:
            try:
                models[model_name] = KeyedVectors.load(kv_path)
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
        else:
            print(f"Skipping {model_name} due to insufficient file size.")
    return models

# Define the directory containing your .kv models
models_dir = '/mnt/c/TopicModelling/word embedding/third round lemma/'

# Load the models
models = load_keyed_vectors_from_directory(models_dir)

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the provided pre-trained model
provided_model_path = "/mnt/c/TopicModelling/word embedding/pre trained models/allLASLA-lemmi-fast-100-SKIP-win5-min5.vec" #example among the different models to be chosen
provided_model = KeyedVectors.load_word2vec_format(provided_model_path)

target_word = 'rex'  # The target word for similarity comparison

models_300d = {name: model for name, model in models.items() if 'dim300' in name}

# Iterate over filtered models to compare with the provided pre-trained model
for model_name, model in models_300d.items():
    if target_word in model.key_to_index and target_word in provided_model.key_to_index:
        embedding_model = model[target_word]
        # The provided model has 100 dimensions, slice to compare same dimensionality
        embedding_provided_model = provided_model[target_word]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_model[:100]], [embedding_provided_model])
        
        print(f"Cosine similarity between '{target_word}' in {model_name} and the provided model: {similarity[0][0]:.4f}")
    else:
        print(f"The word '{target_word}' is not in the vocabulary of one or both models: {model_name}")