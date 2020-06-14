# msds462_final_project
Code Related to Northwestern University MSDS 462 Final Project

<b>Executive Summary</b><br>
Project was to generate an edge-based Object Detection model that would be deployed through an iOS application that I can use on my iPhone. 

My first pass was to attempt utilizing a Faster R-CNN model, trained on Open Image datasets (https://github.com/openimages/dataset_)

I was able to get through the preprocessing of images and train a model, but had issues with converting that to an MLModel file that can be used in iOS.

Thus, to complete the project, I used the TuriCreate API (https://apple.github.io/turicreate/docs/api/) to obtain and process images, then train an object detection model and deploy it in an MLModel file for use in CoreML.

That process was successful and the app was able to detect the presence of cars and bicycles, which not falsely hitting on skateboarders when tested in the real world.

<b>Final Report: </b>https://github.com/jvdowd/msds462_final_project/blob/master/dowd_final_project.pdf<br><br>
<b>Index of Notebooks and Code</b><br><br>
<b>Preprocessing Notebook for Faster R-CNN:  </b>https://github.com/jvdowd/msds462_final_project/blob/master/Dowd_MSDS%20462_Final_Project_Faster%20RCNN%20Test%20Code.ipynb<br>
<br>
<b>Training Notebook for Faster R-CNN:  </b>https://github.com/jvdowd/msds462_final_project/blob/master/Dowd_MSDS%20462_Final%20Project_Faster%20RCNN%20Training%20Code.ipynb<br>
<br>
<b>Test Notebook for Faster R-CNN:  </b>https://github.com/jvdowd/msds462_final_project/blob/master/Dowd_MSDS%20462_Final_Project_Faster%20RCNN%20Test%20Code.ipynb<br><br>
<b>Image Acquisiton and Prep for TuriCreate Model:  </b>https://github.com/jvdowd/msds462_final_project/blob/master/msds462_final_project_ig02_images_data_preparation.ipynb<br>
<br>
<b>TuriCreate Model Training Code:  </b>https://github.com/jvdowd/msds462_final_project/blob/master/msds462_final_project_object_detection_model_training_code_Turi.ipynb<br>
