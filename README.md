# Final-Project-Group9
## Machine Learning 2
Jason Achonu, Kyi Win, Manoah Farmer 

Link to Bucket Containing Dataset: https://storage.googleapis.com/group8project2jkm/Dataset.zip

Use “wget https://storage.googleapis.com/group8project2jkm/Dataset.zip” to download the data set and unzip the folder.

Download the data setThe download.py script is used by first making the directory called “images” using the command “mkdir images” followed by running “python download.py --urls photos/photos.txt” outside of the images directory. This downloads the images into a folder called “images”. 

Next run the code to split the data into categories by running “python split_image_train_test.py”. 

After splititng we ran the the cnn model code using “python Final_cuda.py”. Be sure to run this outside of the images and photos directory.
