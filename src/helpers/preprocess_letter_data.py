# -*- coding: UTF-8 -*-

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for LETTER format")
    parser.add_argument("--dataset", type=str, default="Beauty", help="Dataset name")
    parser.add_argument("--input_path", type=str, default="../datasets/raw", help="Input raw data path")
    parser.add_argument("--output_path", type=str, default="../datasets/processed", help="Output processed data path")
    parser.add_argument("--rating_threshold", type=float, default=3.0, help="Rating threshold for positive samples")
    parser.add_argument("--min_interactions", type=int, default=5, help="Minimum interactions per user")
    parser.add_argument("--min_items", type=int, default=5, help="Minimum interactions per item")
    return parser.parse_args()


def load_amazon_data(input_path, dataset):
    """
    Load Amazon dataset with rating and timestamp information.
    """
    print(f"Loading Amazon {dataset} dataset...")
    
    # Load interactions
    inter_file = os.path.join(input_path, f"{dataset}.csv")
    if not os.path.exists(inter_file):
        print(f"Warning: {inter_file} not found, trying alternative format...")
        # Try alternative file names
        for alt_name in [f"{dataset}_5.csv", f"{dataset}_interactions.csv"]:
            alt_file = os.path.join(input_path, alt_name)
            if os.path.exists(alt_file):
                inter_file = alt_file
                break
        else:
            raise FileNotFoundError(f"Could not find interaction file for {dataset}")
    
    # Read interactions
    df = pd.read_csv(inter_file)
    
    # Standardize column names
    column_mapping = {
        'user_id': 'user_id',
        'item_id': 'item_id', 
        'rating': 'rating',
        'timestamp': 'timestamp',
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'overall': 'rating',
        'unixReviewTime': 'timestamp'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Filter by rating threshold
    df = df[df['rating'] > args.rating_threshold]
    
    print(f"After rating filtering (> {args.rating_threshold}): {len(df)} interactions")
    
    return df


def load_yelp_data(input_path, dataset):
    """
    Load Yelp dataset with rating and timestamp information.
    """
    print(f"Loading Yelp dataset...")
    
    # Load interactions
    inter_file = os.path.join(input_path, "yelp_academic_dataset_review.json")
    if not os.path.exists(inter_file):
        raise FileNotFoundError(f"Could not find Yelp interaction file: {inter_file}")
    
    # Read JSON file
    interactions = []
    with open(inter_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            interactions.append({
                'user_id': data['user_id'],
                'item_id': data['business_id'],
                'rating': data['stars'],
                'timestamp': data['date']
            })
    
    df = pd.DataFrame(interactions)
    
    # Convert timestamp to unix timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
    
    # Filter by rating threshold
    df = df[df['rating'] > args.rating_threshold]
    
    print(f"After rating filtering (> {args.rating_threshold}): {len(df)} interactions")
    
    return df


def filter_data(df, min_interactions=5, min_items=5):
    """
    Filter data by minimum interactions per user and item.
    """
    print("Filtering data by minimum interactions...")
    
    # Filter users with minimum interactions
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(valid_users)]
    
    # Filter items with minimum interactions
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_items].index
    df = df[df['item_id'].isin(valid_items)]
    
    print(f"After filtering: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    
    return df


def create_remapped_data(df):
    """
    Create remapped user and item IDs starting from 0.
    """
    print("Creating remapped IDs...")
    
    # Create user and item mappings
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
    
    # Apply mappings
    df['user_id'] = df['user_id'].map(user_id_map)
    df['item_id'] = df['item_id'].map(item_id_map)
    
    print(f"Remapped to {len(unique_users)} users and {len(unique_items)} items")
    
    return df, user_id_map, item_id_map


def create_absolute_timestamp_split(df, train_ratio=0.7, valid_ratio=0.1):
    """
    Split data by absolute timestamp.
    """
    print("Creating absolute timestamp split...")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate split points
    total_interactions = len(df)
    train_end = int(total_interactions * train_ratio)
    valid_end = int(total_interactions * (train_ratio + valid_ratio))
    
    # Split data
    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]
    
    print(f"Split: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    
    return train_df, valid_df, test_df


def create_user_sequences(df):
    """
    Create user interaction sequences sorted by timestamp.
    """
    print("Creating user sequences...")
    
    user_sequences = defaultdict(list)
    
    # Group by user and sort by timestamp
    for user_id, group in df.groupby('user_id'):
        sorted_group = group.sort_values('timestamp')
        user_sequences[user_id] = sorted_group['item_id'].tolist()
    
    return dict(user_sequences)


def save_letter_format(user_sequences, output_path, dataset):
    """
    Save data in LETTER format (JSON files).
    """
    print("Saving in LETTER format...")
    
    # Create output directory
    output_dir = os.path.join(output_path, dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save interactions
    inter_file = os.path.join(output_dir, f"{dataset}.inter.json")
    with open(inter_file, 'w') as f:
        json.dump({str(uid): items for uid, items in user_sequences.items()}, f)
    
    # Create tokenized index file (LETTER format with <a_xxx>, <b_xxx>, <c_xxx>, <d_xxx>)
    index_file = os.path.join(output_dir, f"{dataset}.index.json")
    max_item_id = max(max(items) for items in user_sequences.values()) if user_sequences else 0
    
    # Generate tokenized indices in LETTER format
    tokenized_indices = {}
    for i in range(max_item_id + 1):
        # Generate 4-token representation like LETTER
        token_a = np.random.randint(0, 256)
        token_b = np.random.randint(0, 256) 
        token_c = np.random.randint(0, 256)
        token_d = np.random.randint(0, 256)
        tokenized_indices[str(i)] = [f"<a_{token_a}>", f"<b_{token_b}>", f"<c_{token_c}>", f"<d_{token_d}>"]
    
    with open(index_file, 'w') as f:
        json.dump(tokenized_indices, f)
    
    # Create item features file (for compatibility)
    item_file = os.path.join(output_dir, f"{dataset}.item.json")
    item_features = {str(i): {"title": f"Item {i}", "description": f"Description for item {i}"} 
                     for i in range(max_item_id + 1)}
    with open(item_file, 'w') as f:
        json.dump(item_features, f)
    
    print(f"Saved LETTER format files to {output_dir}")
    print(f"Generated {len(tokenized_indices)} tokenized item representations")


def main():
    global args
    args = parse_args()
    
    print(f"Processing {args.dataset} dataset...")
    
    # Load data based on dataset type
    if args.dataset.lower() == 'yelp':
        df = load_yelp_data(args.input_path, args.dataset)
    else:
        df = load_amazon_data(args.input_path, args.dataset)
    
    # Filter data
    df = filter_data(df, args.min_interactions, args.min_items)
    
    # Create remapped IDs
    df, user_id_map, item_id_map = create_remapped_data(df)
    
    # Create absolute timestamp split
    train_df, valid_df, test_df = create_absolute_timestamp_split(df)
    
    # Create user sequences for each split
    train_sequences = create_user_sequences(train_df)
    valid_sequences = create_user_sequences(valid_df)
    test_sequences = create_user_sequences(test_df)
    
    # Combine all sequences (LETTER uses all data together)
    all_sequences = {}
    for sequences in [train_sequences, valid_sequences, test_sequences]:
        all_sequences.update(sequences)
    
    # Save in LETTER format
    save_letter_format(all_sequences, args.output_path, args.dataset)
    
    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()
