# Dokumentasi Aplikasi Klasifikasi Kematangan Apel Fuji

## 📋 Daftar Isi

1. [Gambaran Umum](#gambaran-umum)
2. [Arsitektur Aplikasi](#arsitektur-aplikasi)
3. [Flow Aplikasi](#flow-aplikasi)
4. [Penjelasan Kode](#penjelasan-kode)
5. [Algoritma Klasifikasi](#algoritma-klasifikasi)
6. [Interface Pengguna](#interface-pengguna)
7. [Dataset dan Preprocessing](#dataset-dan-preprocessing)
8. [Evaluasi dan Validasi](#evaluasi-dan-validasi)

---

## 🎯 Gambaran Umum

Aplikasi ini adalah sistem web untuk mengklasifikasikan tingkat kematangan apel Fuji berdasarkan analisis warna gambar. Sistem menggunakan kombinasi **Rule-based Classification** dan **K-Nearest Neighbor (KNN)** untuk menghasilkan akurasi yang optimal.

### Tujuan Aplikasi:

- Membantu petani/pedagang menentukan kematangan apel Fuji
- Otomatisasi proses sortir berdasarkan tingkat kematangan
- Mengurangi subjektivitas dalam penilaian manual

### Kategori Klasifikasi:

1. **Mentah** (Raw) - Apel yang belum siap konsumsi
2. **Setengah Matang** (Half-ripe) - Apel dalam tahap pematangan
3. **Matang** (Ripe) - Apel siap konsumsi

---

## 🏗️ Arsitektur Aplikasi

```
┌─────────────────────┐
│   Frontend (HTML)   │ ← Interface pengguna
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Flask Backend     │ ← Server web Python
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Image Processing   │ ← OpenCV untuk ekstraksi fitur
│     (OpenCV)        │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Classification    │ ← Rule-based + KNN
│    Algorithm        │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Result Display    │ ← Hasil klasifikasi
└─────────────────────┘
```

---

## 🔄 Flow Aplikasi

### 1. Inisialisasi Sistem

```python
# Memuat dataset dan melatih model KNN
df_total = pd.read_csv('dataset_apel_fuji.csv')
X = df_total[['MeanR', 'MeanG', 'MeanB', 'MeanH', 'MeanS', 'MeanV']]
y = df_total['Kematangan']
```

**Penjelasan**: Sistem dimulai dengan memuat dataset yang berisi 240 gambar apel Fuji (80 per kategori) beserta fitur warna yang sudah diekstraksi sebelumnya.

### 2. Preprocessing Data

```python
# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Penjelasan**:

- Data dibagi 80% untuk training, 20% untuk testing
- Fitur dinormalisasi agar memiliki skala yang sama
- Stratify memastikan setiap kelas terwakili proporsional

### 3. Training Model KNN

```python
knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='distance')
knn.fit(X_train_scaled, y_train)
```

**Penjelasan**:

- Menggunakan 3 tetangga terdekat (k=3)
- Metrik Manhattan untuk menghitung jarak
- Weight='distance' memberi bobot lebih pada tetangga yang lebih dekat

### 4. User Interface Flow

#### a. Halaman Utama (`index.html`)

```html
<button onclick="window.location.href='/klasifikasi'">Mulai Klasifikasi</button>
```

**Penjelasan**: Halaman landing yang menjelaskan tentang apel Fuji dan mengundang pengguna untuk memulai klasifikasi.

#### b. Halaman Klasifikasi (`klasifikasi.html`)

```html
<form action="/klasifikasi" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/*" required />
  <button type="submit">KLASIFIKASI</button>
</form>
```

**Penjelasan**: Form upload gambar dengan validasi untuk menerima hanya file gambar.

### 5. Proses Klasifikasi

#### a. Upload dan Validasi File

```python
@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Simpan file ke folder uploads
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
```

#### b. Ekstraksi Fitur Warna

```python
def extract_features(img_path):
    # Baca gambar
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ekstraksi fitur RGB (0-1)
    mean_r = np.mean(img_rgb[:, :, 0]) / 255.0
    mean_g = np.mean(img_rgb[:, :, 1]) / 255.0
    mean_b = np.mean(img_rgb[:, :, 2]) / 255.0

    # Konversi ke HSV dan ekstraksi fitur
    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_h = np.mean(hsv_img[:, :, 0])        # Hue (0-180)
    mean_s = np.mean(hsv_img[:, :, 1]) / 255.0  # Saturation (0-1)
    mean_v = np.mean(hsv_img[:, :, 2]) / 255.0  # Value (0-1)

    return [mean_r, mean_g, mean_b, mean_h, mean_s, mean_v]
```

**Penjelasan Detail**:

- **RGB**: Representasi warna dalam Red, Green, Blue
- **HSV**: Hue (warna), Saturation (intensitas), Value (kecerahan)
- **Normalisasi**: Mengubah nilai 0-255 menjadi 0-1 untuk konsistensi
- **Mean**: Rata-rata nilai piksel untuk setiap channel warna

#### c. Algoritma Klasifikasi Hybrid

```python
def classify_apple(features):
    mean_r, mean_g, mean_b, mean_h, mean_s, mean_v = features

    # Rule-based classification (berdasarkan analisis data)
    if mean_g > 0.960 and mean_s < 0.080:
        return 1  # Mentah
    elif mean_g < 0.910 or (mean_g < 0.920 and mean_s > 0.085):
        return 3  # Matang
    elif 0.930 <= mean_g <= 0.940 and mean_s > 0.080:
        return 2  # Setengah matang
    else:
        # Fallback ke KNN untuk kasus yang tidak tertangkap rule
        features_scaled = scaler.transform([features])
        prediction = knn.predict(features_scaled)[0]
        return prediction
```

**Penjelasan Algoritma**:

1. **Rule-based**: Aturan yang dikembangkan berdasarkan analisis dataset
2. **KNN Fallback**: Jika tidak memenuhi rule, gunakan KNN
3. **Logika Rule**:
   - Apel mentah: Hijau tinggi (>0.960), saturasi rendah (<0.080)
   - Apel matang: Hijau rendah (<0.910) atau kombinasi hijau-saturasi
   - Apel setengah matang: Nilai hijau sedang dengan saturasi cukup

### 6. Output dan Visualisasi

```python
labels = {1: 'Mentah', 2: 'Setengah Matang', 3: 'Matang'}
result = labels[prediction]

return render_template('klasifikasi.html',
                     result=result,
                     file=file)
```

**Penjelasan**: Hasil numerik dikonversi ke label yang mudah dipahami dan ditampilkan bersama gambar yang diupload.

---

## 🧠 Algoritma Klasifikasi

### 1. Feature Extraction (Ekstraksi Fitur)

#### RGB Color Space

```python
mean_r = np.mean(img_rgb[:, :, 0]) / 255.0  # Red channel
mean_g = np.mean(img_rgb[:, :, 1]) / 255.0  # Green channel
mean_b = np.mean(img_rgb[:, :, 2]) / 255.0  # Blue channel
```

**Mengapa RGB?**

- Representasi warna yang intuitif
- Green channel paling sensitif terhadap perubahan kematangan
- Mudah diinterpretasi secara visual

#### HSV Color Space

```python
hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
mean_h = np.mean(hsv_img[:, :, 0])        # Hue (warna)
mean_s = np.mean(hsv_img[:, :, 1]) / 255.0  # Saturation (intensitas)
mean_v = np.mean(hsv_img[:, :, 2]) / 255.0  # Value (kecerahan)
```

**Mengapa HSV?**

- Hue merepresentasikan warna murni (tidak terpengaruh pencahayaan)
- Saturation menunjukkan intensitas warna
- Value menunjukkan kecerahan keseluruhan

### 2. Rule-based Classification

#### Analisis Karakteristik Setiap Kelas:

**Apel Mentah**:

- Warna hijau dominan (MeanG > 0.960)
- Saturasi rendah (MeanS < 0.080)
- Belum ada perubahan warna ke merah/kuning

**Apel Matang**:

- Warna hijau berkurang (MeanG < 0.910)
- Mulai muncul warna merah/kuning
- Saturasi meningkat karena perubahan warna

**Apel Setengah Matang**:

- Warna hijau sedang (0.930-0.940)
- Saturasi cukup (>0.080)
- Transisi antara mentah dan matang

### 3. K-Nearest Neighbor (KNN)

#### Parameter KNN:

- **k=3**: Jumlah tetangga terdekat
- **metric='manhattan'**: Jarak Manhattan (L1)
- **weights='distance'**: Bobot berdasarkan jarak

#### Formula Jarak Manhattan:

```
distance = |x1-x2| + |y1-y2| + |z1-z2| + ...
```

#### Proses KNN:

1. Hitung jarak fitur input dengan semua data training
2. Pilih 3 tetangga terdekat
3. Beri bobot berdasarkan jarak (yang dekat bobotnya besar)
4. Tentukan kelas berdasarkan voting berbobot

---

## 🎨 Interface Pengguna

### 1. Halaman Utama (`index.html`)

#### Fitur-fitur:

- **Informasi Edukasi**: Penjelasan tentang apel Fuji
- **Call-to-Action**: Tombol untuk memulai klasifikasi
- **Responsive Design**: Dapat diakses dari berbagai perangkat

#### CSS Styling:

```css
.gradient-background {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### 2. Halaman Klasifikasi (`klasifikasi.html`)

#### Komponen Utama:

- **File Upload**: Input untuk gambar apel
- **Image Preview**: Pratinjau gambar yang diupload
- **Result Display**: Menampilkan hasil klasifikasi
- **Navigation**: Tombol kembali ke halaman utama

#### JavaScript Interaktivity:

```javascript
// Preview gambar sebelum upload
fileInput.addEventListener("change", function (e) {
  const reader = new FileReader();
  reader.onload = function (ev) {
    imgPreview.src = ev.target.result;
    imgPreview.classList.remove("hide");
  };
  reader.readAsDataURL(this.files[0]);
});
```

---

## 📊 Dataset dan Preprocessing

### Struktur Dataset:

```
dataset_apel_fuji.csv:
- Gambar: nama file
- MeanR, MeanG, MeanB: fitur RGB
- MeanH, MeanS, MeanV: fitur HSV
- Kematangan: label (1=Mentah, 2=Setengah Matang, 3=Matang)
```

### Distribusi Data:

- **Total**: 240 gambar
- **Mentah**: 80 gambar (33.3%)
- **Setengah Matang**: 80 gambar (33.3%)
- **Matang**: 80 gambar (33.3%)

### Preprocessing Steps:

1. **Normalisasi**: Fitur RGB dan SV dibagi 255
2. **Standardisasi**: Z-score normalization untuk KNN
3. **Train-Test Split**: 80%-20% dengan stratifikasi

---

## 🔍 Evaluasi dan Validasi

### Metrik Evaluasi:

1. **Accuracy**: Persentase prediksi yang benar
2. **Precision**: Ketepatan per kelas
3. **Recall**: Kemampuan mendeteksi per kelas
4. **F1-Score**: Harmonic mean precision dan recall

### Validasi Model:

```python
# Evaluasi KNN
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi KNN: {accuracy:.4f}")
```

### Kelebihan Sistem:

1. **Hybrid Approach**: Kombinasi rule-based dan machine learning
2. **Interpretable**: Aturan yang mudah dipahami
3. **Robust**: Fallback ke KNN untuk kasus edge
4. **Fast**: Ekstraksi fitur dan klasifikasi yang cepat

### Limitasi:

1. **Pencahayaan**: Sensitif terhadap kondisi cahaya
2. **Variasi Apel**: Terbatas pada apel Fuji
3. **Background**: Asumsi background tidak mempengaruhi
4. **Kualitas Gambar**: Memerlukan gambar dengan kualitas baik

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Persiapan Environment:

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi:

```bash
python app.py
```

### 3. Akses Web Interface:

```
http://localhost:5001
```

---

## 📝 Kesimpulan

Aplikasi klasifikasi kematangan apel Fuji ini menggunakan pendekatan hybrid yang menggabungkan rule-based classification dengan machine learning (KNN). Sistem ini dirancang untuk:

1. **Praktis**: Interface yang mudah digunakan
2. **Akurat**: Kombinasi dua metode untuk hasil optimal
3. **Cepat**: Proses klasifikasi real-time
4. **Scalable**: Dapat dikembangkan untuk varietas apel lain

Dengan dokumentasi ini, diharapkan dapat memberikan pemahaman yang komprehensif tentang cara kerja sistem secara keseluruhan untuk presentasi kepada dosen pembimbing.
