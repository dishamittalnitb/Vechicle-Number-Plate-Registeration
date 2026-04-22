from roboflow import Roboflow

rf = Roboflow(api_key="XMQYEWlxV3KYIQj4AFVc")  

project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(4).download("yolov8")

print("Dataset downloaded!")
print("Dataset location:", dataset.location)

# This complete dataset is stored in the "License-Plate-Recognition-4" folder. 
# You can find the images and annotations in the respective subfolders within that directory. 