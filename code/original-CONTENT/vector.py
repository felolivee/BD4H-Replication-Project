import gensim
from patient_data_reader import PatientReader
from config import Config

# Set up the data reader
FLAGS = Config()
data_sets = PatientReader(FLAGS)

# Get training data
X_raw_data, _ = data_sets.get_data_from_type("train")

# Prepare sentences for word2vec
sentences = []
for patient in X_raw_data:
    for visit in patient:
        # Each code is treated as a "word"
        sentences.append([str(code) for code in visit])

# Train word2vec model
model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save vectors in the format expected by the CONTENT code
with open("../resource/word2vec.vector", "w") as f:
    # Write header: vocab_size and dimension
    f.write(f"{len(model.wv)} 100\n")

    # Write each vector
    for word in model.wv.key_to_index:
        word_id = int(word)  # Convert back to integer
        vector_str = " ".join([str(val) for val in model.wv[word]])
        f.write(f"{word_id} {vector_str}\n")