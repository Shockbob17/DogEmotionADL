# Dog Emotion Classification with Multi‑Level Attention
This is the repository for 60.001 Applied Deep Learning Dog Emotion Classification with Multi‑Level Attention
The report can be found here: https://drive.google.com/file/d/1DQV4sCy37yGuy_9951pU1skLvk8TFsNS/view?usp=sharing 

## Setup
### Python Version
Our project was done on Python **3.10.16**


### Library Dependcies
Run the following code to download all required libraries for the first setup

```
git clone https://github.com/Shockbob17/DogEmotionADL.git
cd dogemotionadl
pip install -r requirements.txt
```

## Dataset
### Unprocessed Dataset
Our team used the dataset from https://www.kaggle.com/datasets/danielshanbalico/dog-emotion.

This dataset can also be downloaded as a zip from https://drive.google.com/file/d/15vCDXS-3GtNHxgL4EczcxMAGQDfVCYe5/view?usp=sharing.

### Processed Dataset
Our team processed the dataset using the following augmentations:

- Remove black and white images
- Zoom into dog face in each image and ensure all images are 456 * 456 using Yolo V8 
- Random horizontal flips & random rotation up to max 10 degrees—simulating head tiltz
- Colour Jitter such as modifying the brightness, contrast, saturation, and hue of images—boost contrast invariance


This processed dataset can  be downloaded as a zip from https://drive.google.com/file/d/1XhSO100qgRuLEyopfb7-4gBp0CRjZkfg/view?usp=drive_link

## Running the code
### modelDemo.ipynb 
To view the results of the different model architectures, please use this notebook.

Our projects assume that the directory the notebook will be run in has the following structure
```
ROOT
├── models
│   ├── *model*
│   │   └── EfficientNetB5.ipynb
│   └── .. 
├── notebooks
│   ├── ..
│   └── EfficientNetB5.ipynb
└── input
    └── final_split_15Apr2025
        ├── train
        ├── eval
        └── test
* if dataset or models are not downloaded, dataset and model would download in the notebook
```


### *models*.ipynb 
To train your own models, please run the different notebooks in the `/notebooks` folder. *(with the exception of DataPreparation.ipynb and modelDemo.ipynb)*

Our team has the following models with the following structures:
| Base Model      | Architectural Additions               | Attention Type                                      |
|-----------------|--------------------------------------|----------------------------------------------------|
| DINOv2          | None                                 | Global Self-Attention                              |
| Swin Transformer| None                                 | Hierarchical Local Attention                       |
| EfficientNetB5  | None                                 | None                                               |
| EfficientNetB5  | SE + ASF                             | Channel-Wise Attention                             |
| EfficientNetB5  | SE + ASF + DINOv2 Head               | Channel-Wise Attention + Spatial Weight Attention + Global Self-Attention |
| EfficientNetB5  | SE + ASF + Swin Transformer Head     | Channel-Wise Attention + Spatial Weight Attention + Hierarchical Local Attention |

**Assumed project structure:**

Our projects assume that the directory the notebook will be run in has the following structure
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

### DataPreparation.ipynb 
Should you wish to reprocess and split the raw dataset, you can choose to run this notebook. It will create the same directory as if you were to download and extract the dataset using the code in the main model notebooks.

Note: You do not need to run the dataset_creation.ipynb as the dataset has already been curated
```
├── notebooks
│   ├── DataPreparation.ipynb    <- Notebook to create the dataset 
│   └── ..                       <- Various notebooks to train different models
```

## File Structure
```
DOGEMOTIONADL
├─── input                          <- folder for the dataset
│
├─── logs                           <- folder containing the trainer logs of the models       
│
├─── models                         <- folder containing the various models        
│
├─── notebooks                      <- folder containing  notebooks of the various models
│    ├─── DataPreparation.ipynb     <- notebook used for dataset creation
│    ├─── DINOv2.ipynb
│    ├─── EfficientNetB5.ipynb
│    ├─── EfficientNetDINOv2.ipynb
│    ├─── ENSE.ipynb
│    ├─── ENSESwin.ipynb
│    ├─── modelDemo.ipynb
│    └─── Swin.ipynb
│
├─── utils                          <- folder containing util functions 
│
└─── results                        <- folder containing results from testing various models
```
