# ME5071_Final_proj
## Installation
```
pip install -r requrements.txt
pip install -e .
```
## Dataset
```
https://huggingface.co/datasets/t8015ny/ME5071_Final_Proj/resolve/main/source_dir.tar.gz?download=true
```
## Model
```
https://huggingface.co/datasets/t8015ny/ME5071_Final_Proj/resolve/main/best_model.pth?download=true
```

## Project structure example
```
ME5071_Final_proj
|--DAMASKRCNN/
|--data/
|--source_dir/    <- source data
   |--cabbages
   |--white_cabbage
|--src/
|--runs/
|--tools/
|--requirements.txt
|--setup.py
```

## Training
```
python tools/train_damaskrcnn.py
```

## Evaluating
```
python tools/evaluate.py --weight path to weight --data_root path to data root
```
Please note that test images should be put in ./data/test/images/ and labels should be in ./data/test/labels/
All labels shoud be in YOLO format.


## For YOLO format evaluate v2
```
python tools/evaluate.py --weight path to weight --data_root path to data root
```
example:
```
python tools/evaluate.py --weight path to weight --data_root data/test/hard/
```
