"""
Module for preprocessing data for machine learning tasks.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os


def preprocess_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess tabular data.

    Args:
        df (pd.DataFrame): The input DataFrame containing tabular data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with selected features and appropriate data types.
    """

    # Drop specific columns
    df = df.drop(['feature_8', 'feature_22'], axis=1) # Poor variability, observed in EDA

    # Drop columns with more than 99% missing values
    df = df.dropna(thresh=df.shape[0] * 0.01, axis=1)

    # Change to correct data types
    categorical_features = [
        'source_id', 'feature_1', 'feature_2', 
        'feature_58', 'feature_59', 'feature_64', 
        'feature_77', 'feature_75', 'feature_76'
    ]
    
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')

    # Drop columns with correlation > 0.95
    df_numeric = df.select_dtypes(include=[np.number])
    correlation = df_numeric.corr()
    correlation_abs = correlation.abs()
    upper_triangle = correlation_abs.where(np.triu(np.ones(correlation_abs.shape), k=1).astype(bool))
    columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    df = df.drop(columns_to_drop, axis=1)

    return df


def load_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model from Hugging Face.

    Args:
        model_name (str): The name or path of the SentenceTransformer model to be loaded.

    Returns:
        SentenceTransformer: The loaded SentenceTransformer model.
    """

    model = SentenceTransformer(model_name)
    
    return model



def add_text_img_embedding(df: pd.DataFrame, model_name: str = "clip-ViT-B-16") -> pd.DataFrame:
    """Add text and image embeddings to the input DataFrame.

    This function adds text and image embeddings to the input dataframe. Both embeddings are obtained using 
    a ViT model from sentence-transformers and Huggingface. This model is chosen because it is SoTA for 
    multimodal tasks and it is able to process both text and images. Moreover we only need to load 1 model 
    in memory, which is an advantage when working with large models together with large datasets. As most 
    descriptions and images can be repeated, the embeddings are stored in a dictionary to avoid recalculating them.

    Args:
        df (pd.DataFrame): The input DataFrame containing descriptions for which 
                           embeddings need to be generated.
        model_name (str): The name of the model to be loaded from Hugging Face. 
                          Defaults to "clip-ViT-B-16".

    Returns:
        pd.DataFrame: A DataFrame with added columns for text and image embeddings, 
                      where each column corresponds to a dimension of the embeddings.
    """

    # Load model
    model = load_model(model_name)

    df['description'] = df['description'].astype('string')

    # Create dictionaries to store embeddings
    text_embeddings_dict = {}
    image_embeddings_dict = {}

    # List of image extensions to try when loading images
    image_extensions = ['.jpg', '.jpeg', '.png']

    # Embedding size
    embedding_size = 512

    # Iterate through the descriptions and images to calculate the embeddings
    for description in set(df["description"]):

        # Calculate the embedding for the description
        text_embeddings_dict[description] = model.encode(description)

        # Transform image file
        image_filename = description.lower().replace(" ", "_").replace("-", "_")
        
        # Try to load the image using different extensions
        image = None
        for ext in image_extensions:
            image_path = f"data/spacecraft_images/{image_filename}{ext}"
            if os.path.exists(image_path):
                image = Image.open(image_path)
                break

        # If the image is not found, try to load the image without dashes
        if image is None:
            image_filename_no_dash = description.lower().replace(" ", "_").replace("-", "")
            for ext in image_extensions:
                image_path = f"data/spacecraft_images/{image_filename_no_dash}{ext}"
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    break

        # If the image is found, calculate its embedding
        if image is not None:
            image_embeddings_dict[description] = model.encode(image)
        else:
            print(f"Image not found: {description}")
            # If the image is not found, add a None to the dictionary
            image_embeddings_dict[description] = [None] * embedding_size

    # Add embeddings to DataFrame
    embedding_dim = len(text_embeddings_dict[list(text_embeddings_dict.keys())[0]])
    for i in range(embedding_dim):
        df[f"text_embedding_{i + 1}"] = df["description"].apply(lambda x: text_embeddings_dict[x][i])
        df[f"image_embedding_{i + 1}"] = df["description"].apply(lambda x: image_embeddings_dict[x][i])
    
    df = df.drop(columns=['description'])
    
    return df
