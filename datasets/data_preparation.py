
import os
import cv2 as cv

from configs import u_net_config


def prepare_dataset():
    dataset_path=u_net_config.test_dataset_path
    mask_path=u_net_config.test_mask_path

    image_list=os.listdir(dataset_path)



    #read image one by one
    for image_name in range(0,len(image_list)):

        path=os.path.join(dataset_path,image_list[image_name])
        print(path)
        image=cv.imread(path)


        #now convert it to binary image using thresholding
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        retval, threshold = cv.threshold(grayImage, 50, 255, cv.THRESH_BINARY)

        mask_image_path=os.path.join(mask_path,image_list[image_name])
        cv.imwrite(mask_image_path,threshold)



if __name__ == '__main__':
    prepare_dataset()
