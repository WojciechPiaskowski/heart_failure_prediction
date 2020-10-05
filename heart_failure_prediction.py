# imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# style options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 30)
sns.set_style('whitegrid')

# load in the data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',', header=0)

# first look at the data
df.head()
df.describe()