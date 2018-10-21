import dlib

if __name__ == "__main__":

    training_xml = "/home/naoki/dlib-19.1/tools/imglab/build/main/mouth_detector9_train.xml"
    testing_xml = "/home/naoki/dlib-19.1/tools/imglab/build/main/mouth_detector9_test.xml"
    svm_file = "/home/naoki/dlib-19.1/tools/imglab/build/main/mouth_detector9.svm"
    options = dlib.simple_object_detector_training_options()
    # 学習時に画像の左右の反転を行うか
    options.add_left_right_image_flips = True
    # コストパラメータ
    options.C = 5
    # スレッド数
    options.num_threads = 4
    # 学習中に詳細を表示するか
    options.be_verbose = True
    
    dlib.train_simple_object_detector(training_xml, svm_file, options)

    print("")
    print("Training accuracy :{}".format(dlib.test_simple_object_detector(training_xml, svm_file)))
    print("Testing accuracy :{}".format(dlib.test_simple_object_detector(testing_xml, svm_file)))


