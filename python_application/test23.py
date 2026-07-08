from preprocessing import preprocess

file = "data/training_data/monolayer/07102025_1.txt"

data = preprocess(file)

print(data.head())
print(data.shape)

