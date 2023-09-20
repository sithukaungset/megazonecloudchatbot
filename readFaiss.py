import faiss

# Define the path to the .faiss file
index_path = "dbs/documentation/faiss_index/index.faiss"

# Load the index
try:
    index = faiss.read_index(index_path)
    print("Index loaded successfully")

except Exception as e:
    print(f"An error occurred: {e}")

# Print basic properties of the index
print(f"Number of vectors in the index: {index.ntotal}")
print(f"Dimension of the vectors: {index.d}")

if hasattr(index, "xb"):
    vectors = index.xb
    # To see the first vector, for instance:
    print(vectors[0])
else:
    print("This index type does not allow direct access to stored vectors.")


import numpy as np

# Create a random query vector
query_vector = np.random.rand(1, index.d).astype('float32')

# Search the index
D, I = index.search(query_vector, 10)  # Retrieve top 10 closest vectors

print(f"Indices of the closest vectors: {I}")
print(f"Distances to the closest vectors: {D}")



# From here we can perform operations on the index as needed.