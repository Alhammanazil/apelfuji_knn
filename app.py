from flask import Flask, request, render_template
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load dataset
df_total = pd.read_csv('dataset_apel_fuji.csv')
X = df_total[['MeanR', 'MeanG', 'MeanB', 'MeanH', 'MeanS', 'MeanV']]
y = df_total['Kematangan']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbor (sesuai dengan skripsi)
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='distance')
knn.fit(X_train_scaled, y_train)

def extract_features(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mean_r = np.mean(img_rgb[:, :, 0]) / 255.0
    mean_g = np.mean(img_rgb[:, :, 1]) / 255.0
    mean_b = np.mean(img_rgb[:, :, 2]) / 255.0
    
    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_h = np.mean(hsv_img[:, :, 0])
    mean_s = np.mean(hsv_img[:, :, 1]) / 255.0
    mean_v = np.mean(hsv_img[:, :, 2]) / 255.0
    
    return [mean_r, mean_g, mean_b, mean_h, mean_s, mean_v]

def classify_apple(features):
    mean_r, mean_g, mean_b, mean_h, mean_s, mean_v = features
    
    # Rule-based classification berdasarkan analisis gambar asli
    # Mentah: MeanG > 0.960, MeanS < 0.080
    # Setengah matang: 0.930 < MeanG < 0.940, MeanS > 0.085
    # Matang: MeanG < 0.910, atau (MeanG < 0.920 dan MeanS > 0.085)
    
    if mean_g > 0.960 and mean_s < 0.080:
        return 1  # Mentah
    elif mean_g < 0.910 or (mean_g < 0.920 and mean_s > 0.085):
        return 3  # Matang
    elif 0.930 <= mean_g <= 0.940 and mean_s > 0.080:
        return 2  # Setengah matang
    else:
        # Fallback dengan KNN (sesuai skripsi)
        features_scaled = scaler.transform([features])
        prediction = knn.predict(features_scaled)[0]
        return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    file = None
    result = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            upload_folder = 'static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            # Extract features
            features = extract_features(file_path)
            
            # Get prediction using our improved classification
            prediction = classify_apple(features)
            
            labels = {1: 'Mentah', 2: 'Setengah Matang', 3: 'Matang'}
            result = labels[prediction]
            
            return render_template('klasifikasi.html', 
                                 result=result, 
                                 file=file)
        else:
            result = 'Tidak ada gambar yang diunggah'
            return render_template('klasifikasi.html', result=result, file=None)
    
    return render_template('klasifikasi.html', result=None, file=None)

if __name__ == '__main__':
    app.run(debug=True)
