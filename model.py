import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load intents from JSON file
with open('intents.json') as file:
    data = json.load(file)
    intents = data['intents']

# Extract intent patterns and responses
patterns = []
responses = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        responses.append(intent['responses'][0])
        tags.append(intent['tag'])

# Encode intents into numerical form
label_encoder = LabelEncoder()
tags_encoded = label_encoder.fit_transform(tags)

# Convert patterns to lowercase
patterns = [pattern.lower() for pattern in patterns]

# Tokenize patterns (split them into words)
word_tokens = [pattern.split() for pattern in patterns]

# Flatten list of tokens
all_words = [word for tokens in word_tokens for word in tokens]

# Create vocabulary set and remove duplicates
vocab = sorted(set(all_words))

# Create a dictionary that maps words to integers
word_to_int = {word: i for i, word in enumerate(vocab)}

# Define function to vectorize input patterns
def vectorize_pattern(pattern):
    vector = np.zeros(len(vocab))
    for word in pattern.split():
        if word in word_to_int:
            vector[word_to_int[word]] = 1
    return vector

# Vectorize input patterns
X = np.array([vectorize_pattern(pattern) for pattern in patterns])

# One-hot encode the encoded tags
y = to_categorical(tags_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save('first_aid_chatbot_model.h5')

print("Model trained, evaluated, and saved.")
