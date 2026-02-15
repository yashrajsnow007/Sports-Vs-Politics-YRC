"""
Step 1: Data Collection - Download AG News dataset
The AG News dataset contains news articles in 4 categories:
1 - World (Politics)
2 - Sports
3 - Business
4 - Sci/Tech

We'll extract only Sports (2) and World/Politics (1)
"""

import urllib.request
import csv
import os
import random

def download_ag_news():

    # AG News dataset URLs (hosted on various sources)
    # Using a common mirror
    train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    
    os.makedirs("data", exist_ok=True)
    
    print("Downloading AG News training data")
    try:
        urllib.request.urlretrieve(train_url, "data/ag_news_train.csv")
        print("  Downloaded: data/ag_news_train.csv")
    except Exception as e:
        print(f"  Error downloading train data: {e}")
        return None, None
    
    print("Downloading AG News test data")
    try:
        urllib.request.urlretrieve(test_url, "data/ag_news_test.csv")
        print("  Downloaded: data/ag_news_test.csv")
    except Exception as e:
        print(f"  Error downloading test data: {e}")
        return None, None
    
    return "data/ag_news_train.csv", "data/ag_news_test.csv"

# 
def load_and_filter_data(filepath, categories={1: 'politics', 2: 'sports'}):
    """
    Load AG News CSV and filter for specific categories.
    
    AG News CSV format: "class_index","title","description"
    Class indices: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
    
    Returns:
        list of tuples: (text, label)
    """
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                class_idx = int(row[0])
                title = row[1]
                description = row[2]
                
                # Filter only politics (1) and sports (2)
                if class_idx in categories:
                    # Combine title and description
                    text = f"{title} {description}"
                    label = categories[class_idx]
                    data.append((text, label))
    
    return data

def create_dataset_files(train_data, test_data, samples_per_class=500):
    """
    Create separate files for sports and politics data.
    Also create combined train/test files.
    """
    os.makedirs("data", exist_ok=True)
    
    # Separate by class
    train_sports = [t for t, l in train_data if l == 'sports']
    train_politics = [t for t, l in train_data if l == 'politics']
    test_sports = [t for t, l in test_data if l == 'sports']
    test_politics = [t for t, l in test_data if l == 'politics']
    
    # Shuffle and limit samples for manageable size
    random.seed(42)
    random.shuffle(train_sports)
    random.shuffle(train_politics)
    random.shuffle(test_sports)
    random.shuffle(test_politics)
    
    train_sports = train_sports[:samples_per_class]
    train_politics = train_politics[:samples_per_class]
    test_sports = test_sports[:samples_per_class // 5]  # 20% of train size
    test_politics = test_politics[:samples_per_class // 5]
    
    # Save individual class files
    with open("data/sports_train.txt", 'w', encoding='utf-8') as f:
        for text in train_sports:
            f.write(text.replace('\n', ' ') + '\n')
    
    with open("data/politics_train.txt", 'w', encoding='utf-8') as f:
        for text in train_politics:
            f.write(text.replace('\n', ' ') + '\n')
    
    with open("data/sports_test.txt", 'w', encoding='utf-8') as f:
        for text in test_sports:
            f.write(text.replace('\n', ' ') + '\n')
    
    with open("data/politics_test.txt", 'w', encoding='utf-8') as f:
        for text in test_politics:
            f.write(text.replace('\n', ' ') + '\n')
    
    print(f"\nDataset created:")
    print(f"  Training: {len(train_sports)} sports, {len(train_politics)} politics")
    print(f"  Testing:  {len(test_sports)} sports, {len(test_politics)} politics")
    print(f"\nFiles saved in 'data/' folder:")
    print(f"  - sports_train.txt")
    print(f"  - politics_train.txt")
    print(f"  - sports_test.txt")
    print(f"  - politics_test.txt")
    
    return {
        'train_sports': train_sports,
        'train_politics': train_politics,
        'test_sports': test_sports,
        'test_politics': test_politics
    }

def analyze_dataset(dataset):
    """
    Basic dataset analysis for the report.
    """
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    all_sports = dataset['train_sports'] + dataset['test_sports']
    all_politics = dataset['train_politics'] + dataset['test_politics']
    
    # Word count statistics
    def get_stats(texts):
        word_counts = [len(text.split()) for text in texts]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        return avg_words, min_words, max_words
    
    sports_stats = get_stats(all_sports)
    politics_stats = get_stats(all_politics)
    
    print(f"\nSports articles ({len(all_sports)} total):")
    print(f"  Avg words per article: {sports_stats[0]:.1f}")
    print(f"  Min words: {sports_stats[1]}, Max words: {sports_stats[2]}")
    
    print(f"\nPolitics articles ({len(all_politics)} total):")
    print(f"  Avg words per article: {politics_stats[0]:.1f}")
    print(f"  Min words: {politics_stats[1]}, Max words: {politics_stats[2]}")
    
    # Most common words (simple frequency)
    def get_common_words(texts, top_n=15):
        word_freq = {}
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'and', 'or', 'that', 'this', 'with', 'as', 
                     'by', 'from', 'it', 'be', 'has', 'have', 'had', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'do', 'does', 'did',
                     'been', 'being', 'its', 'his', 'her', 'their', 'they', 'he', 
                     'she', 'we', 'you', 'i', 'my', 'your', 'our', 'but', 'if', 'so',
                     'not', 'no', 'all', 'any', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very',
                     's', 't', 'just', 'also', 'after', 'before', 'over', 'under',
                     'again', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'what', 'which', 'who', 'whom', 'while', 'although'}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if word and word not in stopwords and len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    print("\nMost common words in SPORTS:")
    for word, count in get_common_words(all_sports):
        print(f"  {word}: {count}")
    
    print("\nMost common words in POLITICS:")
    for word, count in get_common_words(all_politics):
        print(f"  {word}: {count}")
    
    # Sample articles
    print("\n" + "-" * 60)
    print("SAMPLE ARTICLES")
    print("-" * 60)
    
    print("\nSample SPORTS article:")
    print(f"  \"{all_sports[0][:200]}...\"")
    
    print("\nSample POLITICS article:")
    print(f"  \"{all_politics[0][:200]}...\"")


def main():
    print("=" * 60)
    print("  STEP 1: DATA COLLECTION - AG News Dataset")
    print("=" * 60)
    print()
    
    # Download dataset
    train_path, test_path = download_ag_news()
    
    if not train_path:
        print("Failed to download dataset. Creating sample data instead...")
        # Fallback: create sample data manually
        create_sample_data()
        return
    
    # Load and filter data
    print("\nLoading and filtering data (Sports & Politics only)...")
    train_data = load_and_filter_data(train_path)
    test_data = load_and_filter_data(test_path)
    
    print(f"  Train: {len(train_data)} articles")
    print(f"  Test: {len(test_data)} articles")
    
    # Create dataset files
    dataset = create_dataset_files(train_data, test_data, samples_per_class=500)
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    print("\n" + "=" * 60)
    print("Step 1 complete! Data saved in 'data/' folder.")
    print("Next: Step 2 - Feature Extraction")
    print("=" * 60)


def create_sample_data():
    """Fallback: Create sample data if download fails."""
    os.makedirs("data", exist_ok=True)
    
    # Sample sports headlines/articles
    sports_samples = [
        "Lakers defeat Celtics in overtime thriller as LeBron scores 40 points",
        "Manchester United signs new striker for record transfer fee",
        "Serena Williams announces retirement from professional tennis",
        "World Cup final draws record television audience worldwide",
        "Olympic committee announces new sports for 2028 games",
        # ... add more
    ]
    
    # Sample politics headlines/articles  
    politics_samples = [
        "Senate passes new infrastructure bill with bipartisan support",
        "President meets with foreign leaders at G20 summit",
        "Supreme Court rules on landmark voting rights case",
        "Congress debates new immigration reform legislation",
        "Prime Minister announces cabinet reshuffle amid controversy",
        # ... add more
    ]
    
    with open("data/sports_train.txt", 'w') as f:
        f.write('\n'.join(sports_samples))
    
    with open("data/politics_train.txt", 'w') as f:
        f.write('\n'.join(politics_samples))
    
    print("Sample data created in 'data/' folder")


if __name__ == "__main__":
    main()
