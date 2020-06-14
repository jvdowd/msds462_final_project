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
<br>
<br>
<b>References</b><br><br>
Apple. “Recognizing Objects in Live Capture.” Recognizing Objects in Live Capture | Apple Developer Documentation, developer.apple.com/documentation/vision/recognizing_objects_in_live_capture.<br><br>
Apple. “Turi Create API Documentation¶.” Turi Create API Documentation - Turi Create API 6.3 Documentation, apple.github.io/turicreate/docs/api/.<br><br>
Hennon, Yan. “Yhenon/Keras-Rcnn.” GitHub, 25 Aug. 2017, github.com/yhenon/keras-rcnn.<br><br>
Ren, Shaoqing, et al. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, 2017, pp. 1137–1149., doi:10.1109/tpami.2016.2577031.<br><br>
Tryolabs. “Faster R-CNN: Down the Rabbit Hole of Modern Object Detection.” Tryolabs Blog, Tryolabs, 18 Jan. 2018, tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/.<br><br>
Xu, Yinghan. “Faster R-CNN (Object Detection) Implemented by Keras for Custom Data from Google's Open Images...” Medium, Towards Data Science, 25 Feb. 2019, towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a.<br><br>
Xu, Yinghan. “RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras.” GitHub, 8 Apr. 2020, github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras.<br><br>

