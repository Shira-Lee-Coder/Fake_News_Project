import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import config

def load_and_preprocess_data():
    """
    Load data and perform preprocessing
    Return: X_train, X_test, y_train, y_test, vectorizer
    """
    print(f"Loading data file: {config.DATA_FILE} ...")
    
    # 1. Read data
    try:
        df = pd.read_csv(config.DATA_FILE)
    except FileNotFoundError:
        print(f"Error: File not found {config.DATA_FILE}, please confirm that the file is in the current folder.")
        return None

    # 2. Check if the column name exists
    if config.TEXT_COLUMN not in df.columns or config.LABEL_COLUMN not in df.columns:
        print(f"Error: Column not found '{config.TEXT_COLUMN}' or '{config.LABEL_COLUMN}' in CSV file.")
        print(f"The current file contains the following columns: {df.columns.tolist()}")
        return None

    # 3. Remove null values (to prevent errors)
    df = df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN])
    
    # 4. Separate features (X) and labels (y)
    X = df[config.TEXT_COLUMN].astype(str) 
    y = df[config.LABEL_COLUMN]

    # 5. Text vectorization (Core step: converting text into numbers)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=config.MAX_FEATURES
    )

    X_vectorized = vectorizer.fit_transform(X)

    # 6. Divide the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, 
        y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )

    print(f"Data preparation is complete! Training set: {X_train.shape[0]}. Test set: {X_test.shape[0]}.")
    
    return X_train, X_test, y_train, y_test, vectorizer