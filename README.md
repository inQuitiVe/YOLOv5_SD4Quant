!git clone https://github.com/inQuitiVe/YOLOv5_SD4Quant.git  # clone \\
%cd YOLOv5_SD4Quant \\
!pip install -qr requirements.txt
!pip install torchsummary
!conda install -c 3dhubs gcc-5
!pip install Ninja
!pip install wandb
!python trainQ.py  --img 640 --batch 16 --epochs 5000 --data coco128.yaml --weights '' --cfg yolov5s.yaml   --sync-bn
!python trainQ.py  --img 640 --batch 16 --epochs 5000 --data coco128.yaml --weights runs/train/<your checkpoint>/weights/last.pt --cfg yolov5s.yaml --cache --sync-bn 
