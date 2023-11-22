import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from normalisasi import *

# Load your dataset
df = pd.read_csv('TikTok_reviews_with_sentiment.csv')

texts = df['Review'].values
labels = df['Sentiment'].values

train_reviews = texts[:35000]
train_sentiments = labels[:35000]


norm_train_reviews = pre_process_corpus(train_reviews)

# Tokenize and preprocess the data
tokenizer = Tokenizer(oov_token='<UNK>')
tokenizer.fit_on_texts(norm_train_reviews)
tokenizer.word_index['<PAD>'] = 0
max([(k, v) for k, v in tokenizer.word_index.items()], key = lambda x:x[1]), min([(k, v) for k, v in tokenizer.word_index.items()], key = lambda x:x[1]), t.word_index['<UNK>']
sequences = tokenizer.texts_to_sequences(norm_train_reviews)
X = pad_sequences(sequences)

print("Vocabulary size={}".format(len(tokenizer.word_index)))
print("Number of Documents={}".format(tokenizer.document_count))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=X.shape[1]))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Display and save the classification report
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print('Classification Report:')
print(classification_rep)

# Export the classification report to a CSV file
classification_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)).transpose()
classification_df.to_csv('classification_report.csv', index=True)
