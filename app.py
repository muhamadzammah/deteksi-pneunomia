import os
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
from db_config import init_db, db

# -------------------------------------------------------
# Konfigurasi Aplikasi Flask
# -------------------------------------------------------
app = Flask(__name__)
app.secret_key = "secret_key"

# Inisialisasi database
db = init_db(app)

# Direktori upload
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

# -------------------------------------------------------
# Load Model Machine Learning
# -------------------------------------------------------
knn = joblib.load('models/knn_model.joblib')
scaler = joblib.load('models/scaler.joblib')
le = joblib.load('models/labels.joblib')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

# -------------------------------------------------------
# Model Database
# -------------------------------------------------------
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100))
    umur = db.Column(db.Integer)
    alamat = db.Column(db.String(200))
    jenis_kelamin = db.Column(db.String(10))
    file_path = db.Column(db.String(200))
    predicted_label = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(50))
    uploaded_at = db.Column(db.DateTime, server_default=db.func.now())

# -------------------------------------------------------
# Fungsi bantu
# -------------------------------------------------------
def extract_feature_from_pil(pil_img):
    pil_img = pil_img.convert('RGB').resize(IMG_SIZE)
    arr = image.img_to_array(pil_img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    out = feature_model.predict(arr, verbose=0)[0]
    pooled = out.mean(axis=(0, 1))
    return pooled.reshape(1, -1)


def severity_from_confidence(confidence, predicted_label):
    label_lower = predicted_label.strip().lower()

    if "pneumonia" in label_lower:
        if confidence < 0.6:
            return "Ringan", [
                "Lakukan observasi mandiri di rumah selama 3–5 hari.",
                "Perbanyak istirahat dan konsumsi air putih minimal 2 liter per hari.",
                "Konsumsi makanan bergizi tinggi protein dan vitamin C.",
                "Hindari merokok dan paparan asap.",
                "Jika demam atau batuk menetap lebih dari 5 hari, segera konsultasikan ke dokter paru."
            ]
        elif confidence < 0.85:
            return "Sedang", [
                "Segera periksa ke dokter umum atau spesialis paru untuk pemeriksaan fisik dan rontgen lanjutan.",
                "Kemungkinan memerlukan antibiotik oral sesuai anjuran dokter.",
                "Pantau suhu tubuh dan saturasi oksigen.",
                "Batasi aktivitas berat dan istirahat cukup.",
                "Minum obat pereda demam bila suhu tubuh >38°C."
            ]
        else:
            return "Berat", [
                "Segera ke IGD atau rumah sakit terdekat.",
                "Kemungkinan diperlukan rawat inap.",
                "Lakukan pemeriksaan darah lengkap dan foto toraks.",
                "Ikuti seluruh instruksi dokter."
            ]
    else:
        return "Normal", [
            "Paru-paru tampak normal berdasarkan hasil analisis citra.",
            "Jaga kebersihan udara di rumah dan hindari polusi.",
            "Olahraga ringan secara rutin.",
            "Konsumsi buah dan sayur.",
            "Lakukan pemeriksaan kesehatan rutin setiap 6–12 bulan."
        ]

# -------------------------------------------------------
# Fungsi Visualisasi Preprocessing Lengkap
# -------------------------------------------------------
def preprocessing_visualization(pil_img, filename):
    import cv2
    from PIL import ImageOps, Image

    save_dir = os.path.join('static', 'processed')
    os.makedirs(save_dir, exist_ok=True)

    # --- Tahap 1: Citra RGB (Asli) ---
    rgb_path = os.path.join(save_dir, f"rgb_{filename}")
    pil_img = pil_img.convert('RGB')
    pil_img.save(rgb_path)

    # Konversi ke array grayscale dasar
    gray_img = ImageOps.grayscale(pil_img)
    gray_array = np.array(gray_img)

    # --- Tahap 2: Histogram Equalization (Peningkatan Kontras) ---
    equalized_array = cv2.equalizeHist(gray_array)
    equalized_img = Image.fromarray(equalized_array)
    equalized_path = os.path.join(save_dir, f"equalized_{filename}")
    equalized_img.save(equalized_path)

    # --- Tahap 3: Citra Biner (Thresholding Otsu) ---
    _, binary_array = cv2.threshold(equalized_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = Image.fromarray(binary_array)
    binary_path = os.path.join(save_dir, f"binary_{filename}")
    binary_img.save(binary_path)

    # --- Tahap 4: Gaussian Blur (Reduksi Noise) ---
    blur_array = cv2.GaussianBlur(equalized_array, (5, 5), 0)
    blur_img = Image.fromarray(blur_array)
    blur_path = os.path.join(save_dir, f"blur_{filename}")
    blur_img.save(blur_path)

    # --- Tahap 5: Morphological Opening (Membersihkan Noise Kecil) ---
    kernel = np.ones((3, 3), np.uint8)
    opened_array = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, kernel)
    opened_img = Image.fromarray(opened_array)
    opened_path = os.path.join(save_dir, f"opened_{filename}")
    opened_img.save(opened_path)

    # --- Tahap 6: Edge Detection (Canny) ---
    edges_array = cv2.Canny(equalized_array, 50, 150)
    edges_img = Image.fromarray(edges_array)
    edges_path = os.path.join(save_dir, f"edges_{filename}")
    edges_img.save(edges_path)

    # Samakan ukuran untuk tampilan (224x224)
    for path in [rgb_path, equalized_path, binary_path, blur_path, opened_path, edges_path]:
        img = Image.open(path).resize((224, 224))
        img.save(path)

    # Kembalikan semua path
    return {
        "rgb": rgb_path,
        "equalized": equalized_path,
        "binary": binary_path,
        "blur": blur_path,
        "opened": opened_path,
        "edges": edges_path
    }


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            nama = request.form['nama']
            umur = request.form['umur']
            alamat = request.form['alamat']
            jenis_kelamin = request.form['jenis_kelamin']
            file = request.files['file']

            if not file or file.filename == '':
                flash("⚠️ Harap unggah file X-Ray terlebih dahulu.")
                return redirect(url_for('index'))

            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            pil_img = Image.open(save_path)

            # 🔹 Buat hasil tahapan preprocessing (tanpa disimpan ke DB)
            preprocessed_paths = preprocessing_visualization(pil_img, filename)

            # 🔹 Ekstraksi fitur dan prediksi
            feat = extract_feature_from_pil(pil_img)
            feat_scaled = scaler.transform(feat)
            probs = knn.predict_proba(feat_scaled)[0]
            idx_pred = np.argmax(probs)
            label = le.inverse_transform([idx_pred])[0]
            confidence = float(probs[idx_pred])
            severity, recommendations = severity_from_confidence(confidence, label)

            # 🔹 Simpan data pasien ke database (tanpa gambar preprocessing)
            new_patient = Patient(
                nama=nama,
                umur=umur,
                alamat=alamat,
                jenis_kelamin=jenis_kelamin,
                file_path=f"static/uploads/{filename}",
                predicted_label=label,
                confidence=confidence,
                severity=severity
            )
            db.session.add(new_patient)
            db.session.commit()

            # 🔹 Tampilkan hasil di halaman
            return render_template(
                'index.html',
                result=True,
                nama=nama,
                umur=umur,
                alamat=alamat,
                jenis_kelamin=jenis_kelamin,
                label=label,
                confidence=round(confidence * 100, 2),
                severity=severity,
                recommendations=recommendations,
                file_path=f"static/uploads/{filename}",
                preprocessed_paths=preprocessed_paths
            )

        except Exception as e:
            flash(f"❌ Terjadi kesalahan: {str(e)}")
            return redirect(url_for('index'))

    return render_template('index.html', result=False)


@app.route('/admin')
def admin_page():
    patients = Patient.query.order_by(Patient.uploaded_at.desc()).all()
    return render_template('admin.html', data=patients)


# -------------------------------------------------------
# Endpoint API untuk Flutter
# -------------------------------------------------------
@app.route('/predict', methods=['POST'])
def api_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nama file kosong'}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        pil_img = Image.open(save_path)
        feat = extract_feature_from_pil(pil_img)
        feat_scaled = scaler.transform(feat)
        probs = knn.predict_proba(feat_scaled)[0]
        idx_pred = np.argmax(probs)
        label = le.inverse_transform([idx_pred])[0]
        confidence = float(probs[idx_pred])
        severity, recommendations = severity_from_confidence(confidence, label)

        result = {
            "label": label,
            "confidence": confidence,
            "severity": severity,
            "recommendations": recommendations
        }

        # Simpan ke database
        new_patient = Patient(
            nama="Pasien API",
            umur=0,
            alamat="Tidak diketahui",
            jenis_kelamin="-",
            file_path=f"static/uploads/{filename}",
            predicted_label=label,
            confidence=confidence,
            severity=severity
        )
        db.session.add(new_patient)
        db.session.commit()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pasien', methods=['GET'])
def api_get_pasien():
    patients = Patient.query.order_by(Patient.uploaded_at.desc()).all()
    result = [
        {
            "nama": p.nama,
            "umur": p.umur,
            "jenis_kelamin": p.jenis_kelamin,
            "alamat": p.alamat,
            "predicted_label": p.predicted_label,
            "confidence": p.confidence,
            "severity": p.severity,
            "file_path": p.file_path
        } for p in patients
    ]
    return jsonify(result)
@app.route('/delete/<int:id>', methods=['POST'])
def delete_patient(id):
    patient = Patient.query.get_or_404(id)
    try:
        db.session.delete(patient)
        db.session.commit()
        flash(f"✅ Data pasien {patient.nama} berhasil dihapus.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ Gagal menghapus data: {str(e)}", "danger")
    return redirect(url_for('admin_page'))

# -------------------------------------------------------
# Main Program
# -------------------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()

