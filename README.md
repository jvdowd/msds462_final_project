# msds462_final_project
Code Related to Northwestern University MSDS 462 Final Project

<b>Executive Summary</b><br>
Project was to generate an edge-based Object Detection model that would be deployed through an iOS application that I can use on my iPhone. 

My first pass was to attempt utilizing a Faster R-CNN model, trained on Open Image datasets (https://github.com/openimages/dataset_)

I was able to get through the preprocessing of images and train a model, but had issues with converting that to an MLModel file that can be used in iOS.

Thus, to complete the project, I used the TuriCreate API (https://apple.github.io/turicreate/docs/api/) to obtain and process images, then train an object detection model and deploy it in an MLModel file for use in CoreML.

That process was successful and the app was able to detect the presence of cars and bicycles, which not falsely hitting on skateboarders when tested in the real world.

Files with Dowd in front are for the R-CNN experimentation, the msds462 are the TuriCreate.
