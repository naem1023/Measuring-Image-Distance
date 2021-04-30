# git clone https://github.com/OlafenwaMoses/ImageAI
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()

model_path = input('input model path e.g) /yolo.h5: ')

detector.setModelPath( os.path.join(execution_path , model_path))
detector.loadModel()


image_path = input('input images directory: ')

image_names = os.listdir(image_path)
print(image_names)


detected_images = []

for image_name in image_names:
    detections = detector.detectObjectsFromImage(input_image=os.path.join(image_path, image_name), output_image_path=os.path.join(execution_path , '_output_image.jpg'), minimum_percentage_probability=30)

    if detections:
        print(image_name)
        print(detections)
        detected_images.append(detections)

print(len(detected_images)/len(image_names))
