import pandas as pd

"""DATA IMPORT"""

df = pd.read_csv("Dataset/StudentMentalHealth.csv")

"""DATA CLEANING"""

# Change time format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Change name of columns
column_mapping = {
'Choose your gender': 'Gender',
'What is your course?': 'Course',
'Your current year of Study': 'Year study',
'What is your CGPA?': 'CGPA',
'Do you have Depression?': 'Depression',
'Do you have Anxiety?': 'Anxiety',
'Do you have Panic attack?': 'Panic attack',
'Did you seek any specialist for a treatment?': 'Specialist treatment'
}
df.rename(columns=column_mapping, inplace=True) 
# Inplace = True to modifies the original DataFrame in place

# Drop NA values
df.isnull().sum()
df = df.dropna(axis=0)

# Change the format of answer from boolean to integer
answer_mapping = {
'Yes' : 1,
'No' : 0
}
df = df.applymap(lambda x: answer_mapping.get(x,x))

# Change the format of Age into integer
df['Age'] = df['Age'].astype(int)

# Change the format of Year Study from object to integer
df['Year study'] = df['Year study'].str.replace(r'[Yy]ear (\d+)', r'\1', regex=True)
df['Year study'] = df['Year study'].astype(int)

# Count the number of different values of the column
df['CGPA'].value_counts()
df['Year study'].value_counts()
df['Course'].value_counts()
