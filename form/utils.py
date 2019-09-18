import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import cv2
import numpy as np

import os
import sys
import time
import numpy as np
import imgaug
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil

def apply_mask(image, mask, color, alpha=0.5):
    """Apply mask to image."""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """Render ball and player graphics."""
    n_instances = boxes.shape[0]

    if not n_instances:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        
        assert label in ("person", "sports ball")
        if label == "person":
          color = (255, 255, 0)
          image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
          
        else:
          color = (255, 0, 255)
          center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
          radius = max((x2 - x1) // 2, (y2 - y1) // 2)
          image = cv2.circle(image, center, radius, color, 2)
        
        # Pixel-wise mask.
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        
        # Text caption.
        score = scores[i] if scores is not None else None
        caption = '{:.2f}'.format(score) if score else ''
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

def filter_results(results, which_classes=("sports ball")):
  """Filter results for Formalytics-specific features."""

  class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

  # For Formalytics, only care about 'person' and 'sports ball'.
  class_idx = [i for i, n in enumerate(class_names) if n in which_classes]
  
#   # Apply filter.  
#   def filter_candidates(class_ids, scores, which_ids):
#     """Return the index of max score per class in which_ids."""
#     assert len(class_ids) == len(scores)
#     d_val = {}  # values
#     d_ind = {}  # indices
#     for i, x_i in enumerate(class_ids):
#       if x_i not in d_val:
#         d_val[x_i] = scores[i]
#         d_ind[x_i] = i  
#       else:
#         if scores[i] > d_val[x_i]:
#           d_val[x_i] = scores[i]
#           d_ind[x_i] = i

#     # indices to keep.
#     return [v for k, v in d_ind.items() if k in which_ids]
  
#   idx = filter_candidates(results['class_ids'], results['scores'], 
#                           class_idx)
    
  rois = results['rois']
  class_ids = results['class_ids']
  scores = results['scores']
  masks = results['masks']
  
  return rois, masks, class_ids, class_names, scores

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)





# # Subclass default COCO config for video inference (batch size > 1).
# batch_size = 3
# class InferenceConfig(Config):
#     NAME="VideoInf"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = batch_size  # batch size.
#     DETECTION_MIN_CONFIDENCE = 0.6

# config = InferenceConfig()
# config.display()

# #@title Process video frame-by-frame
# capture = cv2.VideoCapture("Test.mov")
# print("Processing " + "Test.mov")

# try:
#     if os.path.exists("Output"):
#       # Clear image cache.
#       shutil.rmtree("Output")
#     os.makedirs("Output")
# except OSError:
#     pass

# frames = []
# frame_count = 0

# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break

#     # Buffer frames into batch to fit on GPU.
#     frame_count += 1
#     frames.append(frame)

#     if len(frames) == batch_size:
#         results = model.detect(frames, verbose=0)
              
#         # Process footage frame-by-frame.
#         for i, (frame, r) in enumerate(zip(frames, results)):         
          
#             # Filter results for Formalytics features.
#             frame = display_instances(frame, *filter_results(r))
            
#             # Write processed frame back to image.
#             name = '{0}.jpg'.format(frame_count + i - batch_size)
#             name = os.path.join("Output", name)
#             cv2.imwrite(name, frame)
#             print('writing to file:{0}'.format(name))
            
#         # Start next batch.
#         frames = []

# capture.release()

# #@title Calculate video FPS
# video = cv2.VideoCapture("Test.mov");

# # Find OpenCV version
# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# if int(major_ver)  < 3 :
#     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
#     print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
# else :
#     fps = video.get(cv2.CAP_PROP_FPS)
#     print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

# video.release();

# #@title Write video
# FPS = 60  #@param

# def make_video(outvid, images=None, fps=FPS, size=None,
#                is_color=True, format="FMP4"):
#     """
#     Create a video from a list of images.
 
#     @param      outvid      output video
#     @param      images      list of images to use in the video
#     @param      fps         frame per second
#     @param      size        size of each frame
#     @param      is_color    color
#     @param      format      see http://www.fourcc.org/codecs.php
#     @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
#     The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
#     By default, the video will have the size of the first image.
#     It will resize every image to this size before adding them to the video.
#     """
#     from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
#     fourcc = VideoWriter_fourcc(*format)
#     vid = None
#     for image in images:
#         if not os.path.exists(image):
#             raise FileNotFoundError(image)
#         img = imread(image)
#         if vid is None:
#             if size is None:
#                 size = img.shape[1], img.shape[0]
#             vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
#         if size[0] != img.shape[1] and size[1] != img.shape[0]:
#             img = resize(img, size)
#         vid.write(img)
#     vid.release()
#     return vid

# import glob
# import os

# # Directory of images to run detection on
# images = list(glob.iglob(os.path.join("Output", '*.*')))

# # Sort the images by integer index
# images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

# outvid = os.path.join("./", "out.mp4")
# make_video(outvid, images, fps=FPS)

# !ls -alh ./videos/