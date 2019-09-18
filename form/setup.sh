git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN/
wget https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
pip3 install imgaug
pip3 install Cython matplotlib opencv-python-headless pyyaml Pillow
pip3 install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'