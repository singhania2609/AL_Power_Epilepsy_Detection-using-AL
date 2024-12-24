import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

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
ica = ICA(n_components=15, random_state=30, max_iter=800)
ica.fit(raw)
ica.exclude = [0, 2]  # Exclude components manually based on visual inspection
raw_cleaned = raw.copy()
ica.apply(raw_cleaned)

# Extract signal data
fs = int(raw.info['sfreq'])
cleaned_signal = raw_cleaned.get_data()  # Shape: (channels, timepoints)

# 2. Prepare Data for Logistic Regression
# Calculate mean and standard deviation for each channel as features
mean_features = np.mean(cleaned_signal, axis=1)  # Shape: (channels,)
std_features = np.std(cleaned_signal, axis=1)    # Shape: (channels,)

# Combine features into a feature matrix
features = np.vstack((mean_features, std_features)).T  # Shape: (channels, 2)

# Generate placeholder labels
labels = np.random.randint(0, 2, size=features.shape[0])  # Binary labels for demonstration

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 3. Train Logistic Regression Model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Epileptic", "Epileptic"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Visualization
# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Epileptic", "Epileptic"])
disp.plot(cmap='viridis')
plt.title('Confusion Matrix')
plt.show()
