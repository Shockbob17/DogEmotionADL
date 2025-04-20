# Dog Emotion Classification with Multi‑Level Attention
This is the repository for 60.001 Applied Deep Learning Dog Emotion Classification with Multi‑Level Attention
The report can be found here: 


dfgdfssg ergberrybersbyvrtsvetavsr b rbv

## Setup
### Python Version
Our project was done on python **3.10.16**


### Library Dependcies
Run the following code to download all required libraries for the first setup

```
git clone https://github.com/Shockbob17/DogEmotionADL.git
cd DOGEMOTIONADL
pip install -r requirements.txt
```

## Dataset
### Unprocessed Dataset
Our team used the dataset from https://www.kaggle.com/datasets/danielshanbalico/dog-emotion.

This dataset can also be downloaded as a zip from https://drive.google.com/file/d/15vCDXS-3GtNHxgL4EczcxMAGQDfVCYe5/view?usp=sharing.

### Processed Dataset
Our team processed the dataset using the following augmentations:

- Remove black and white images
- Zoom into dog face in each image and ensure all images are 224 * 224 using Yolo V8 
- Random horizontal flips & random rotation up to max 10 degrees—simulating head tilt
- Colour Jitter such as modifying the brightness, contrast, saturation, and hue of images—boost contrast invariance


This processed dataset can  be downloaded as a zip from https://drive.google.com/file/d/1XhSO100qgRuLEyopfb7-4gBp0CRjZkfg/view?usp=drive_link

## Running the code
To train your own models, please run the different notebooks in the `/notebooks` folder. *(with the exception of Datapreperation.ipynb)*

Note: You do not need to run the dataset_creation.ipynb as the dataset has already been curated
```
├── notebooks
│   ├── Datapreperation.ipynb    <- Notebook to create the dataset 
│   └── ..                       <- Various notebooks to train different models
```

Our team has the following models with the following structures:


**Assumed project structure:**

Our projects assume that they will be run in the following structure
```
ROOT
├── notebooks
│   ├── ..
│   └── EfficientNetB5.ipynb
└── input
    └── final_split_15Apr2025
        ├── train
        ├── eval
        └── test
* if dataset not downloaded, dataset would download in the loading dataset section.
```

This procesed dataset as mentioned above will then have to be placed in the `/input` folder.

* if dataset not downloaded, dataset would download in the loading dataset section in `Notebooks/${models}.ipynb`, where ${models} is based on the specific file selected.

### DataPreperation.ipynb 
Should you wish to reprocess and split the raw dataset, you can choose to run this notebook. It will create the same directory as if you were to download and extract the dataset using the code in the main model notebooks.



## File Structure
```
DOGEMOTIONADL
├─── input                          <- folder for the dataset
│
├─── models                         <- folder containing the various models        
│
├─── notebooks                      <- folder containing  notebooks of the various models
│    ├───opencv
│    ├───opencv
│    ├───opencv
│    ├───ENSESwin.ipynb
│    ├───ENSE.ipynb
│    ├───utils                      <- folder containing util functions 
│    └───Datapreperation.ipynb      <- notebook used for dataset creation
│
└─── results                        <- folder containing results from testing various models

```
