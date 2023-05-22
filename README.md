# PointNet classification

Push dataset into `data` folder

### Prepare Csv File

```
python prepare_csv.py
```
### Run code
```
python train.py --root_path <path to dataset> --train_path <path to train.csv file> --val_path <path to test.csv file> --save_model_path <path to save model from training>

```

Output model Save at `output`dir