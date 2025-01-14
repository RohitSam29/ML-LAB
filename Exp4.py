import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

def entropy(target_col):
    
    probabilities = target_col.value_counts(normalize=True)
    entropy_val = -sum(probabilities * probabilities.map(lambda p: math.log2(p) if p > 0 else 0))
    return entropy_val

def info_gain(data, feature, target):

    total_entropy = entropy(data[target])
    values = data[feature].value_counts(normalize=True)
    
    weighted_entropy = sum(values[value] * entropy(data[data[feature] == value][target]) for value in values.index)
    information_gain = total_entropy - weighted_entropy
    return information_gain, weighted_entropy


def load_golf_dataset_from_csv(file_path):

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None


def display_dataset_description(df):

    print("\n--- Golf Dataset ---\n")
    print(df.head())
    print("\n--- Dataset Description ---\n")
    print(df.describe(include='all'))

def main():
 
    file_path = 'golf_dataset.csv' 
    df = load_golf_dataset_from_csv(file_path)
    
    if df is None:
        return
    
    display_dataset_description(df)
    
   
    target_column = "Play"
    initial_entropy = entropy(df[target_column])
    print(f"\nInitial Entropy of '{target_column}': {initial_entropy:.4f}\n")
    
   
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    gains = {}
    
    print("\n--- Entropy After Partitioning by Each Feature ---\n")
    for feature in features:
        gain, entropy_after = info_gain(df, feature, target_column)
        gains[feature] = gain
        print(f"Feature: {feature}")
        print(f"  Entropy after split: {entropy_after:.4f}")
        print(f"  Information Gain: {gain:.4f}\n")
    
   
    sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Feature Ranking Based on Information Gain ---\n")
    for idx, (feature, gain) in enumerate(sorted_gains, start=1):
        print(f"{idx}. {feature} (Information Gain: {gain:.4f})")
    
    # Step 5: Visualize Information Gain
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(gains.keys()), y=list(gains.values()), palette="viridis")
    plt.title('Information Gain of Each Feature')
    plt.xlabel('Features')
    plt.ylabel('Information Gain')
    plt.ylim(0, max(gains.values()) + 0.1)
    
    # Annotating bars
    for index, value in enumerate(gains.values()):
        plt.text(index, value + 0.01, f"{value:.4f}", ha='center')
    
    plt.show()

if __name__ == "__main__":
    main()
