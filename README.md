## 1.
Install CLIP model, refer to:https://github.com/openai/CLIP.git.

Then put the train.by, test.by, and Resnet50_stngleMode.py provided by this code repository into the root directory of the CLIP folder, and extract the dataset file: data.rar to ./data folder.

## 2.
run the code:

### python train.py

for train the clip model based on this dataset.

### python test.py 

for testing the clip model.

### python Resnet50_stngleMode.py 

for trian and testing the Resnet50 model.

## requirement

Our pytorch version is 1.9.0, python version is 3.8.
