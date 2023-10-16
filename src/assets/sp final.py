from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
np.set_printoptions(suppress=True)

# Load your model and class names here
model = load_model("C:/Users/Dell/Downloads/keras_model.h5", compile=False)
class_names = open("C:/Users/Dell/Downloads/labels.txt", "r").readlines()

# Initialize the camera (you can adjust the camera index if needed)

url = 'http://192.168.29.37:8080/video'


camera = cv2.VideoCapture(url)

if not camera.isOpened():
    print("Error: Could not open the camera.")
else:
    while True:
        ret, image = camera.read()

        if not ret:
            print("Error: Could not read from the camera.")
            break

        # Resize the image if it's not empty
        if not image.size == 0:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image1 = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow("Webcam Image", image1)

            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1

            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            output=class_name[2:]
            score=str(np.round(confidence_score * 100))[:-2]
            print(output)


        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:
            break
        
    camera.release()
    cv2.destroyAllWindows()
