# Project Overview

This repository contains a machine learning pipeline that processes tabular and multimodal (text and image) data to train predictive models. The pipeline includes data preprocessing, model training, and embedding generation. The models are then fine-tuned (optional) and saved as pickled files for later use.

The project demonstrates the ability to handle both tabular and multimodal data types. It involves training two models:
1. **Tabular Model**: A traditional machine learning model trained on structured, tabular data.
2. **Multimodal Model**: A model that combines both text and image data. The embeddings for text and images are generated beforehand and used as features for training the multimodal model.

The pipeline is designed to be run inside a Docker container for easy execution and reproducibility.

## Project Structure

The project is structured as follows:

```
├── data/
│   ├── spacecraft_images/   # Folder containing images used for the multimodal model
│   └── data.csv             # Tabular data used for the models
├── models/
│   ├── tabular_model.pkl    # Pickle file for the trained tabular model
│   ├── multimodal_model.pkl # Pickle file for the trained multimodal model
│   ├── tabular_grid_search_log.txt # Logs for the tabular model's grid search
│   └── multimodal_grid_search_log.txt # Logs for the multimodal model's grid search
├── notebooks/
│   └── # Jupyter Notebooks for exploratory data analysis (EDA)
├── src/
│   ├── preprocess.py        # Data preprocessing functions
│   ├── models.py            # Model training functions
│   └── __init__.py          # Initialization for the src package
├── Dockerfile               # Dockerfile for containerizing the application
├── README.md                # Main README file for the project
├── requirements.txt         # Python dependencies required for the project
└── main.py                  # Main entry point to execute the pipeline
```


## Requirements

You can install these dependencies by running:
```
pip install -r requirements.txt
```

## How to run

1. **Build the Docker image:**

   First, build the Docker image. This step should take around 1 minute:

   ```
   docker build -t multimodal .
   ```

2. **Run the image in a container:**

    Next, run the image in a container.  Mount output directory so that files are also saved in host machine. This step should take around 5 minutes:

    ```
    docker run -v "$(pwd)/models:/app/models" -it --rm multimodal
    ```

## Output
For faster execution, by default fine tuning will only be performed on the tabular model. The multimodal_fine_tune_grid_search_log 
file contains the logs of the multimodal fine tuning.
If you want to run fine tuning for both models, change the fine_tunning parameter in the multimodal model to True.

At the end of the execution two pickle files with each model and two text files with the grid search 
logs for each model will be stored in the models folder.

## Data
The data folder (including data.csv and the images in spacecraft_images) is not included in this repository due to size constraints. If you'd like to run the pipeline with your own data, please contact me to request the dataset.

## Contact

For questions, collaboration, or to request access to the dataset:

- **Name:** Laura Menéndez  
- **GitHub:** [LauraMeva](https://github.com/LauraMeva)  
- **LinkedIn:** [LauraMenendezVallejo](https://www.linkedin.com/in/lauramenendezvallejo)
