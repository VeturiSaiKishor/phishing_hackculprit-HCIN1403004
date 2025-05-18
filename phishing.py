# Install required packages (Colab usually has these pre-installed, but this ensures it)
!pip install pandas scikit-learn

import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample datasets (replace with your own or expand)
phishing_urls = [
    'http://192.168.1.1/secure-login',
    'http://login.verify-account.com',
    'https://secure-update-info.com/login',
    'http://freegiftcards.com@malicious.com',
    'http://paypal.com.secure-login.com',
]

legitimate_urls = [
    'https://www.google.com',
    'https://www.amazon.com',
    'https://www.facebook.com',
    'https://www.wikipedia.org',
    'https://www.apple.com',
]

# Create DataFrame with labels
phishing_df = pd.DataFrame({'url': phishing_urls, 'label': 1})
legitimate_df = pd.DataFrame({'url': legitimate_urls, 'label': 0})

# Combine datasets
data = pd.concat([phishing_df, legitimate_df], ignore_index=True)

# Feature extraction function
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['count_dots'] = url.count('.')
    features['count_hyphens'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_equal'] = url.count('=')
    features['count_https'] = 1 if url.startswith('https') else 0
    domain = urlparse(url).netloc
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
    features['has_ip'] = 1 if ip_pattern.fullmatch(domain) else 0
    return features

# Extract features for all URLs
feature_list = [extract_features(url) for url in data['url']]
features_df = pd.DataFrame(feature_list)

# Labels
labels = data['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.3, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate model
print(classification_report(y_test, y_pred))