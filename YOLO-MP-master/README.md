# YOLO-MP

Lightweight forest fire detection model.

## Environment
For the required environment, run `
pip install -r requirements.txt`

## Train
- train.py: Configure your dataset path and other  hyperparameters in `train.py` for training
- CLI: Run yolo train `model=yolov8n.pt/$.yaml data=$.yaml epochs=350 imgsz=640 batch=16` for training
## Predict
```shell:copy
yolo predict model=$.pt source='../fire.jpg'
```
## Results
### Comparison experiment results
| Different models       | P      | R      | mAP50  | mAP50-95 | Params/M | FLOPs/G | Latency/ms |
|------------------------|--------|--------|--------|----------|----------|---------|------------|
| Faster R-CNN           | 0.6530 | 0.4347 | 0.4929 | 0.2353   | 41.75    | 91.48   | 92.00      |
| SSD                    | 0.7344 | 0.7011 | 0.7363 | 0.3988   | 23.75    | 136.81  | 45.54      |
| YOLOv5n                | 0.8764 | 0.7619 | 0.8452 | 0.5221   | 2.50     | 7.06    | 5.32       |
| YOLOv8n                | 0.8729 | 0.7751 | 0.8528 | 0.5360   | 3.01     | 8.09    | 4.75       |
| YOLOv9t                | 0.8544 | 0.7535 | 0.8361 | 0.4942   | 2.62     | 10.74   | 17.98      |
| YOLOv10n               | 0.8703 | 0.7616 | 0.8526 | 0.5432   | 2.26     | 6.52    | 6.25       |
| YOLOv11n               | 0.8660 | 0.7832 | 0.8459 | 0.5283   | 2.58     | 6.31    | 6.22       |
| Hyper-YOLO             | 0.8642 | 0.7832 | 0.8552 | 0.5477   | 3.94     | 10.76   | 8.63       |
| YOLOv12n               | 0.8234 | 0.7066 | 0.7626 | 0.3791   | 2.51     | 5.82    | 12.39      |
| YOLOv13n               | 0.7815 | 0.6865 | 0.7219 | 0.3434   | 2.45     | 6.20    | 18.49      |
| RT-DETR                | 0.8641 | 0.7950 | 0.8714 | 0.5482   | 18.83    | 29.68   | 28.80      |
| YOLO-MP                | 0.8757 | 0.8027 | 0.8680 | 0.5478   | 2.07     | 6.02    | 5.48       |

## Dataset
- [Dataset](https://drive.google.com/drive/folders/1ufAopOK2Oe1uBtuLH6JT92zbxtYIHQmv?usp=sharing)




## reference
- https://github.com/ultralytics/ultralytics
