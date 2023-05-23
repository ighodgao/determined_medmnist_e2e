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
  
    export DET_MASTER=<your_desired_master_ip>

  Set up a python virtual environment and activate it:
    
      python3 -m venv ~/.myvenv
      source ~/.myvenv/bin/activate

  Install requirements:
  
      pip install -r requirements.txt

  Submit an experiment:
    
      det e create config.yaml .
  

## Running inference

  Copy the example environment file `.env_example` to `.env `.

    cp .env_example .env
   
  Change `EXPERIMENT_ID` to your experiment ID from your training run:
    
      EXPERIMENT_ID=<your_experiment_ID>
      
  Generate sample images to run inference on using the test dataset:
  
    mkdir sample_images
    python3 generate_sample_test_images.py

  Deploy the Flask server:
    
    python3 deploy.py

  Run inference on an image (replace filename with desired image name):

    curl -X POST -F file=@sample_images/<filename.jpg> http://localhost:5000/predict
    
More info can be found here:https://medmnist.com/
