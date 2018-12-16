import os
import cv2
import scipy.misc
import numpy as np


def create_video(image_folder, video_name, overview_image_name):

    unordered_images = [name for name in os.listdir(image_folder) if name.endswith(".png")]

    num_images = len(unordered_images)
    image_names = []
    for i in range(1, num_images + 1):
        image_names.append("image_" + str(i) + ".png")
    height, width, layers = 600, 600, 3

    # Resize the images
    resized_images = []
    for img_name in image_names:
        img = cv2.imread(os.path.join(image_folder, img_name))
        resized_img = scipy.misc.imresize(img, (height, width))
        resized_images.append(resized_img)

    overview_image = cv2.imread(overview_image_name)
    resized_overview_img = scipy.misc.imresize(overview_image, (height, width))

    final_images = []
    for img in resized_images:
        final_img = np.concatenate([img, resized_overview_img], 1)
        final_images.append(final_img)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 1, (width * 2, height))

    for image in final_images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def create_videos(max_image_id):
    for i in range(1, max_image_id + 1):
        create_video("./images/example_" + str(i),
                     "./videos/video_" + str(i) + ".avi",
                     "/home/dipendra/Downloads/NavDroneLinuxBuild/overview_images/debugImage_" + str(i - 1) + ".png")

create_videos(200)
