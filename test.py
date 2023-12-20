import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

null_values_train = train_data.isnull().sum()
null_values_test = test_data.isnull().sum()
print("Null-värden i train data:\n", null_values_train)
print("\nNull-värden i test data:\n", null_values_test)
print("Script is running!")

