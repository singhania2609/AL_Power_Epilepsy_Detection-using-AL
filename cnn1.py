import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 1. Load EEG Data
data_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(str(data_path) + '/MEG/sample/sample_audvis_raw.fif', preload=True)

# Pick EEG channels only
raw.pick_types(meg=False, eeg=True)

# Apply Bandpass Filter (0.5 to 40 Hz) and Notch Filter (50 Hz)
raw.filter(0.5, 40)
raw.notch_filter(freqs=50)

# Apply ICA for noise cancellation
from mne.preprocessing import ICA
ica = ICA(n_components=15, random_state=42, max_iter=800)
ica.fit(raw)
ica.exclude = [0, 2]  # Exclude components manually based on visual inspection
raw_cleaned = raw.copy()
ica.apply(raw_cleaned)

# Extract signal data
fs = int(raw.info['sfreq'])
cleaned_signal = raw_cleaned.get_data()  # Shape: (channels, timepoints)

# 2. Prepare Data for CNN
# Reshape data into 3D format for CNN (timepoints x channels x 1)
eeg_images = cleaned_signal.T[:, :, np.newaxis]  # Shape: (timepoints, channels, 1)
labels = np.random.randint(0, 2, size=eeg_images.shape[0])  # Placeholder labels for demonstration

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(eeg_images, labels, test_size=0.3, random_state=42)

# Reshape data to include an additional dimension for CNN input
X_train = X_train.reshape((-1, eeg_images.shape[1], 1, 1))  # (samples, channels, 1, 1)
X_test = X_test.reshape((-1, eeg_images.shape[1], 1, 1))

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 3. Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 1), activation='relu', input_shape=(eeg_images.shape[1], 1, 1)),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(0.25),

    Conv2D(64, (3, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 5. Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# 6. Plot Training and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# 7. Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Epileptic", "Epileptic"])
disp.plot(cmap='viridis')
plt.title('Confusion Matrix')
plt.show()
