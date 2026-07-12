import kagglehub

# Download latest version
path = kagglehub.dataset_download("lokeshparab/gutenberg-books-and-metadata-2025")

print("Path to dataset files:", path)