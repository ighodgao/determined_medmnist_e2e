# MedMNIST with Determined

Example of training a computer vision model on the MedMNIST dataset using Determined AI. Currently configured to train a model on the PathMNIST dataset, which is one of 12 datasets offered by the MedMNIST project. 

## Changing the dataset
  #### For training purposes
  1) Replace `data_flag` in `config.yaml` with the corresponding data flag. Full list can be found [here](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).
  2) Download the desired corresponding dataset from [Zenodo](https://zenodo.org/record/6496656). Upload this to your S3 bucket and update the bucket details in `config.yaml`.
  3) Update  `dataset_name ` in `config.yaml` to the name of this file.
  
  #### For inference purposes
  1) Copy the example environment file `.env_example` to `.env `.
  2) Update `DATASET_FLAG` to the same desired `data_flag`.
  
## Training the model

  After setting up Determined, export DET_MASTER to your desired cluster IP:

  Set up a python virtual environment and activate it:

  Submit an experiment:
  

## Running inference

  Generate sample images to run inference on using the test dataset:

  Deploy the Flask server:

  Run inference on an image (replace filename with desired image name):


More info can be found here:
