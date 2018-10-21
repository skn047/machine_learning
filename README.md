# machine_learning
All scripts are about machine learning.
Now there are only scripts about dlib.

You can calculate IOU of your prediction made by your own detector and test xml file created using imglab with my first script.
Just run "python handmade_test.py "your detector path" "your test folder path""in your terminal.
Everytime you run this script you must change your test xml file path.
After runnning the script test images are shown in a window with the detected box in pink and ground truth one in blue.
Then you close the window and you can see the IOU, minimum IOU and average IOU so far.
I'm not sure whether it works with test xml file which has mutiple boxes because I just run it with test xml file which has only single box but maybe it works.

Second one is the scripts that just train your train your model, create svm file and show training and test score.
Maybe if you can't get 1 score among all items, your model will not work as you want.
