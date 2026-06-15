import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/dogs-vs-cats/train/train"
IMG_SIZE = 128

X = []
y = []

print("Loading images...")

cat_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("cat")]
dog_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("dog")]

# 1000 cats + 1000 dogs
cat_files = cat_files[:1000]
dog_files = dog_files[:1000]

files = cat_files + dog_files
np.random.shuffle(files)

for file in tqdm(files):

    path = os.path.join(DATASET_PATH, file)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    X.append(features)

    if file.startswith("cat"):
        y.append(0)
    else:
        y.append(1)

X = np.array(X)
y = np.array(y)

print("Feature Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training HOG + SVM...")

model = LinearSVC(max_iter=10000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_model.pkl")

print("\nModel Saved!")