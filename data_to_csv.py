import pandas as pd

# Convert .data file into .csv 
df = pd.read_csv('data/iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
df.to_csv('data/iris.csv', index=False)