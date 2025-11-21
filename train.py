import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
from tqdm import tqdm

DATA_DIR = "dataset/chest_xray/train"  # gunakan folder train saja
IMG_SIZE = (224, 224)

def list_images_and_labels(root_dir):
    X_paths = []
    y = []
    for label in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, label)
        if not os.path.isdir(full):
            continue
        for fname in os.listdir(full):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                X_paths.append(os.path.join(full, fname))
                y.append(label)
    return X_paths, y

def predict_and_pool(paths, model):
    feats = []
    for p in tqdm(paths, desc="Ekstraksi fitur"):
        img = image.load_img(p, target_size=IMG_SIZE, color_mode='rgb')
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, 0)
        arr = preprocess_input(arr)
        out = model.predict(arr, verbose=0)[0]
        pooled = out.mean(axis=(0,1))  # global average pooling
        feats.append(pooled)
    return np.array(feats)

def main():
    # backbone feature extractor
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    feat_model = Model(inputs=base.input, outputs=base.output)

    print("Menyiapkan data...")
    all_paths, all_labels = list_images_and_labels(DATA_DIR)

    # split data menjadi train / val / test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.1765, stratify=train_labels, random_state=42
    )
    # 0.1765 * 0.85 ≈ 0.15 → total 70/15/15

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    print("Ekstraksi fitur train...")
    X_train = predict_and_pool(train_paths, feat_model)
    print("Ekstraksi fitur val...")
    X_val = predict_and_pool(val_paths, feat_model)
    print("Ekstraksi fitur test...")
    X_test = predict_and_pool(test_paths, feat_model)

    # encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_val = le.transform(val_labels)
    y_test = le.transform(test_labels)

    # scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # parameter grid untuk tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    print("GridSearchCV berjalan...")
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=2)
    grid.fit(X_train_s, y_train)

    print("Best params:", grid.best_params_)
    print("Best score (cv):", grid.best_score_)

    best_knn = grid.best_estimator_

    # evaluasi di validation dan test
    y_val_pred = best_knn.predict(X_val_s)
    print("Val accuracy:", accuracy_score(y_val, y_val_pred))

    y_test_pred = best_knn.predict(X_test_s)
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # simpan model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_knn, 'models/knn_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(le, 'models/labels.joblib')
    print("Model terbaik disimpan ke folder models/")

if __name__ == "__main__":
    main()
