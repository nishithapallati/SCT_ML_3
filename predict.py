import cv2
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt

IMG_SIZE = 128

model = joblib.load("models/svm_model.pkl")

image_path = "images/dog.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit() 

original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

features = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys'
)

prediction = model.predict([features])[0]

label = "Cat" if prediction == 0 else "Dog"

print("Prediction:", label)

plt.imshow(original)
plt.title(f"Prediction: {label}")
plt.axis("off")
plt.show()