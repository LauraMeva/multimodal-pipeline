"""
Code developed for Imnoo's technical assignment by Laura Menéndez.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from src.preprocess import preprocess_tabular, load_model, add_text_img_embedding
from src.models import train_model
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def main():
    """Main function to run the entire data processing and modeling pipeline.
    """

    # Load data
    print("Loading tabular data...")
    df = pd.read_csv('data/data.csv')
    print("Data loaded successfully")
    
    # Preprocess data
    print("Preprocessing tabular data...")
    popped_col = df.pop('description')
    df = preprocess_tabular(df)
    print("Tabular data preprocessed successfully")

    # Train tabular model
    print("Training tabular model...")
    train_model(df, model_type='tabular', fine_tune=True)
    print("Tabular model trained successfully")
    
    # Create embeddings
    print("Creating embeddings...")
    df['description'] = popped_col
    df = add_text_img_embedding(df)
    print("Embeddings created successfully")
    
    # Train multimodal model
    print("Training multimodal model...")
    # Do not fine-tune the multimodal model to save time, even though it would be recommended
    train_model(df, model_type='multimodal', fine_tune=False)
    print("Multimodal model trained successfully")
    
    print("Pipeline executed successfully")



if __name__ == '__main__':
    main()