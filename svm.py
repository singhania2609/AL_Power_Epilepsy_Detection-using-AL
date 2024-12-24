import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
ica = ICA(n_components=15, random_state=42, max_iter=800)
ica.fit(raw)
ica.exclude = [0, 2]  # Exclude components manually based on visual inspection
raw_cleaned = raw.copy()
ica.apply(raw_cleaned)

# Extract signal data
fs = int(raw.info['sfreq'])
cleaned_signal = raw_cleaned.get_data()  # Shape: (channels, timepoints)

# 2. Feature Extraction
def extract_features(eeg_data, fs):
    """Extract frequency-based features from EEG data."""
    freqs, psd = welch(eeg_data, fs, nperseg=fs * 2)
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)

    delta_power = trapezoid(psd[(freqs >= delta_band[0]) & (freqs <= delta_band[1])], freqs[(freqs >= delta_band[0]) & (freqs <= delta_band[1])])
    theta_power = trapezoid(psd[(freqs >= theta_band[0]) & (freqs <= theta_band[1])], freqs[(freqs >= theta_band[0]) & (freqs <= theta_band[1])])
    alpha_power = trapezoid(psd[(freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], freqs[(freqs >= alpha_band[0]) & (freqs <= alpha_band[1])])

    variance = np.var(eeg_data)

    return [delta_power, theta_power, alpha_power, variance]

# Extract features for all channels
features = [extract_features(channel, fs) for channel in cleaned_signal]

# Create DataFrame for Features
feature_labels = ["Delta Power", "Theta Power", "Alpha Power", "Variance"]
features_df = pd.DataFrame(features, columns=feature_labels)
features_df["Channel"] = [f"EEG {i+1}" for i in range(len(features))]
features_df.set_index("Channel", inplace=True)

# Generate Placeholder Labels
true_labels = np.random.randint(0, 2, size=len(features_df))  # Random labels for demonstration
features_df["Epileptic"] = true_labels

# 3. Classification
# Split data into training and testing sets
X = features_df[feature_labels].values
y = features_df["Epileptic"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM Classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)  # Linear kernel
svm.fit(X_train, y_train)

# Predict and Evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM Classifier Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Epileptic", "Epileptic"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 4. Visualization
# Feature Table
print("\nExtracted Features:")
print(features_df)

# Plotting Decision Boundary (PCA for 2D visualization, optional)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100, label="True Labels")
plt.title("PCA of EEG Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Epileptic (1) / Non-Epileptic (0)")
plt.legend()
plt.show()
