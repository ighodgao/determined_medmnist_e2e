# MedMNIST with Determined

Example of training a computer vision model on the MedMNIST dataset using Determined AI. Currently configured to train a model on the PathMNIST dataset, which is one of 12 2-D datasets offered by the MedMNIST project (https://medmnist.com/). 

## Changing the dataset
  #### For training purposes
  1) Replace `data_flag` in `config.yaml` with the name of a MedMNIST subset. The subset names are listed [here](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py), in the `INFO` variable. For example, `pathmnist` is one of the subset names.
  
  #### For inference purposes
  1) Copy the example environment file `.env_example` to `.env `.
  2) Update `DATASET_FLAG` to the same desired `data_flag`.
  
## Training the model

  After setting up Determined, export DET_MASTER to your desired cluster IP:
  
      export DET_MASTER=<your_desired_master_ip>

  Submit an experiment:
    
      det e create config.yaml .
  

## Running inference

  Install requirements:
  
      pip install -r inference_requirements.txt

  Copy the example environment file `.env_example` to `.env `.

      cp .env_example .env
   
  In the `.env` file, change `EXPERIMENT_ID` to your experiment ID from your training run:
    
      EXPERIMENT_ID=<your_experiment_ID>
      
  Generate sample images to run inference on using the test dataset:
  
      mkdir sample_images
      python3 generate_sample_test_images.py

  Deploy the Flask server:
    
      python3 deploy.py

  Run inference on an image (replace filename with desired image name):

      curl -X POST -F file=@sample_images/<filename.jpg> http://localhost:5000/predict
    

