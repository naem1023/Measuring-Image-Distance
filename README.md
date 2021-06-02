# Measuring-Image-Distance
이미지 데이터(2D) 내에서의 거리 측정, 졸업프로젝트

# Training requirement
- pytorch >= 1.8
- requirements.txt

# Test Predictor
Run test predictor with 'demo.jpg'.
```
# cd service/worker
# python test_predict.py
```

# Test Evaluation
```
# cd service/worker
# python eval.py
```

# Train
- Open train_in_colab.ipynb in colab.
- Using TPU, run from first cell.
- Using GPU, run from 'Train distance via colab'.
- Specify your '*.mat' file in google drive and set path.

# About API server and worker
Read service/README.md
