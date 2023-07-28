### MALLET visualisation: here we provide only one example template for linear plot visualisation we produced ###

#preprocessing: corpus chunking into files of 100 tokens each

import os
import re
from itertools import count

# Set the paths to the input corpus directory and the output directory
corpus_dir = '/mnt/c/TopicModelling/corpus/ita/clean2'
output_dir = '/mnt/c/TopicModelling/mallet/corpus/ita'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each file in the corpus directory
for file_name in os.listdir(corpus_dir):
    file_path = os.path.join(corpus_dir, file_name)

    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into tokens (assuming whitespace tokenization)
    tokens = text.split()

    # Chunk the tokens into smaller chunks of 100 tokens each
    chunk_size = 100
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Generate the output file names with a progressing counting number
    file_base_name = os.path.splitext(file_name)[0]
    file_name_pattern = re.compile(rf'{file_base_name}_(\d+)\.txt')
    existing_numbers = [int(file_name_pattern.match(f).group(1)) for f in os.listdir(output_dir) if file_name_pattern.match(f)]
    file_number = next(count(start=0, step=1 + len(chunks)))
    while file_number in existing_numbers:
        file_number += 1

    # Write each chunk to a separate output file
    for i, chunk in enumerate(chunks):
        output_file_name = f'{file_base_name}_{file_number + i}.txt'
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(' '.join(chunk))

#mallet sub-routine run

import subprocess
import os

mallet_path = '/mnt/c/mallet2/bin/mallet'
input_file = '/mnt/c/mallet2/corpus_dir/input_file.txt'

for topics in range(10, 101, 10):
    for iterations in range(100, 1001, 100):
        output_dir = f'output/{topics}_{iterations}'
        os.makedirs(output_dir, exist_ok=True)

        # Run MALLET
        subprocess.run([mallet_path, 'import-file', '--input', input_file, '--output', f'{output_dir}/corpus.mallet', '--keep-sequence'])

        # Train topics
        try:
            result = subprocess.run([
                mallet_path, 'train-topics', 
                '--input', f'{output_dir}/corpus.mallet', 
                '--num-topics', str(topics), 
                '--num-iterations', str(iterations), 
                '--output-model', f'{output_dir}/model.mallet', 
                '--output-doc-topics', f'{output_dir}/doctopics.txt', 
                '--output-topic-keys', f'{output_dir}/topickeys.txt', 
                '--word-topic-counts-file', f'{output_dir}/wordtopiccounts.txt',
                '--evaluator-filename', f'{output_dir}/evaluator.mallet', 
                '--topic-word-weights-file', f'{output_dir}/wordweights.txt',
                '--diagnostics-file', f'{output_dir}/diagnostics.xml'
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))
        except subprocess.CalledProcessError as e:
            print("Error occurred:")
            print(e.stderr.decode('utf-8'))


#visualisation: template for linear plots

#genre - love-joy topics cluster

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure

df = pd.read_csv ("/mnt/c/TopicModelling/mallet/ita_experiment/topic_distributrion_genre.csv")

figure(figsize = (120,14))
plt.xticks(np.arange(0, 5209, 10)) #sets xtick spacing with min, max and step
plt.xticks(fontsize=16)
plt.xticks(rotation=90)
plt.xlabel('Texts', fontsize=40) #set x axis labels
plt.ylabel('Topic strength', fontsize=40) #set y axis labels
plt.axvline(x = 3643, color = 'black', linewidth=4) #adds vertical line 1
plt.text(300, 0.68, 'Tragedy', fontsize = 30) #set custom legend text 1
plt.text(3950, 0.68, 'Comedy', fontsize = 30) #set custom legend text 2
x = df["names"]
y = df["topic0"] + df["topic3"] + df["topic11"] + df["topic23"]
plt.plot(x, y, label="love-joy topics") #plot x axis + first y axis
plt.legend(loc="upper left", fontsize="20") #set legend

#boxplot visualisations template

#boxplot genres acts distributions: grief-joy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

df = pd.read_csv("/mnt/c/TopicModelling/mallet/lat_experiment/doc_topics_medians_genres_by_act_summed_topics.csv")
plt.figure(figsize = (50,14))
plt.xticks(fontsize=16)
plt.xticks(rotation=90)
plt.rcParams['lines.markersize'] = 30
plt.axvline(x = 4.5, color = 'black', linewidth=4)
plt.axvline(x = 9.5, color = 'black', linewidth=4)
plt.axvline(x = 14.5, color = 'black', linewidth=4)
sns.scatterplot(x = 'medians', y = 'topic5', label="topic5, grief", data = df, color="red")
sns.lineplot(x = 'medians', y = 'topic5', data = df, color="red")
sns.scatterplot(x = 'medians', y = 'topic9', label="topic9, joy", data = df, color="blue")
sns.lineplot(x = 'medians', y = 'topic9', data = df, color="blue")
sns.scatterplot(x = 'medians', y = 'topic12', label="topic12, love", data = df, color="lightblue")
sns.lineplot(x = 'medians', y = 'topic12', data = df, color="lightblue")

#heatmap template
#distribution of topics across genres

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.genfromtxt("/mnt/c/TopicModelling/mallet/doc_topics_medians_full_genres_summed.csv", delimiter=",")
plt.figure(figsize = (18, 9))
xticks = ["christian religion", "kingship-military", "kingship-christian religion", "crime", "family", 
          "grief", "feast", "kingship-religion","christian religion",
          "love","kingship","family", "joy"]
yticks = ["average_comedy", "average_comsac", "average_tragedy", "average_tragsac"]
sns.heatmap(data, linewidths = 0.5, cmap="Blues", xticklabels=xticks, yticklabels=yticks, linecolor='black', vmin=0, vmax=0.4)