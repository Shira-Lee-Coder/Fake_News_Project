import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import config

def generate_news_wordclouds():
    # 1. Load data
    print(f"Reading data file: {config.DATA_FILE}...")
    try:
        df = pd.read_csv(config.DATA_FILE)
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the path.")
        return

    # 2. Basic data cleaning (refer to the logic in data_loader.py)
    df = df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN])
    
    # 3. Split data by tags
    labels = df[config.LABEL_COLUMN].unique()
    num_labels = len(labels)
    fig, axes = plt.subplots(1, num_labels, figsize=(16, 8))
    
    # Set canvas
    fig, axes = plt.subplots(1, num_labels, figsize=(10 * num_labels, 10))

    if num_labels == 1:
        axes = [axes]
    
    # 4. Generate a word cloud
    for i, label_value in enumerate(labels):
        print(f" {label_value}...")
        
        text = " ".join(df[df[config.LABEL_COLUMN] == label_value][config.TEXT_COLUMN].astype(str))
        
        wc = WordCloud(
            width=800, height=800, 
            background_color='white', 
            stopwords=STOPWORDS,
            max_words=100
        ).generate(text)
        
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"Label: {label_value}", fontsize=20)
        axes[i].axis("off")
    
    plt.suptitle("News Word Cloud Comparison Analysis", fontsize=24, y=0.95)
    plt.subplots_adjust(wspace=0.3, top=0.85, bottom=0.1, left=0.1, right=0.9)

    # 5. Save and Display
    output_png = "news_wordcloud_comparison.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Word cloud generated successfully! Saved as: {output_png}")

if __name__ == "__main__":
    generate_news_wordclouds()