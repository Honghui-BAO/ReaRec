#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Test script for LETTER integration with ReaRec.
This script tests the LETTERReader functionality without running the full training pipeline.
"""

import os
import sys
import json
import tempfile
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from helpers.LETTERReader import LETTERReader


def create_test_data(output_dir, dataset_name="TestBeauty"):
    """
    Create test data in LETTER format for testing.
    """
    print(f"Creating test data for {dataset_name}...")
    
    # Create test interactions (user_id -> list of item_ids)
    test_interactions = {
        "0": [0, 1, 2, 3, 4, 5],  # User 0 has 6 interactions
        "1": [1, 2, 3, 4, 5, 6, 7],  # User 1 has 7 interactions
        "2": [2, 3, 4, 5, 6, 7, 8, 9],  # User 2 has 8 interactions
    }
    
    # Create test item indices (item_id -> list of tokens)
    test_indices = {
        "0": ["<a_0>", "<b_1>", "<c_2>"],
        "1": ["<a_1>", "<b_0>", "<c_3>"],
        "2": ["<a_2>", "<b_1>", "<c_4>"],
        "3": ["<a_3>", "<b_2>", "<c_5>"],
        "4": ["<a_4>", "<b_3>", "<c_6>"],
        "5": ["<a_5>", "<b_4>", "<c_7>"],
        "6": ["<a_6>", "<b_5>", "<c_8>"],
        "7": ["<a_7>", "<b_6>", "<c_9>"],
        "8": ["<a_8>", "<b_7>", "<c_0>"],
        "9": ["<a_9>", "<b_8>", "<c_1>"],
    }
    
    # Create test item features
    test_items = {
        "0": {"title": "Test Item 0", "description": "Description for test item 0"},
        "1": {"title": "Test Item 1", "description": "Description for test item 1"},
        "2": {"title": "Test Item 2", "description": "Description for test item 2"},
        "3": {"title": "Test Item 3", "description": "Description for test item 3"},
        "4": {"title": "Test Item 4", "description": "Description for test item 4"},
        "5": {"title": "Test Item 5", "description": "Description for test item 5"},
        "6": {"title": "Test Item 6", "description": "Description for test item 6"},
        "7": {"title": "Test Item 7", "description": "Description for test item 7"},
        "8": {"title": "Test Item 8", "description": "Description for test item 8"},
        "9": {"title": "Test Item 9", "description": "Description for test item 9"},
    }
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Write test files
    with open(os.path.join(dataset_dir, f"{dataset_name}.inter.json"), 'w') as f:
        json.dump(test_interactions, f)
    
    with open(os.path.join(dataset_dir, f"{dataset_name}.index.json"), 'w') as f:
        json.dump(test_indices, f)
    
    with open(os.path.join(dataset_dir, f"{dataset_name}.item.json"), 'w') as f:
        json.dump(test_items, f)
    
    print(f"Test data created in {dataset_dir}")
    return dataset_dir


def test_letter_reader(dataset_dir, dataset_name):
    """
    Test the LETTERReader functionality.
    """
    print(f"Testing LETTERReader with {dataset_name}...")
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.path = os.path.dirname(dataset_dir)
            self.dataset = dataset_name
            self.rating_threshold = 3.0
            self.use_item_features = True
    
    args = MockArgs()
    
    try:
        # Initialize LETTERReader
        reader = LETTERReader(args)
        
        # Test basic properties
        print(f"Number of users: {reader.n_users}")
        print(f"Number of items: {reader.n_items}")
        
        # Test data dictionary structure
        for split in ["train", "valid", "test"]:
            if split in reader.data_dict:
                data = reader.data_dict[split]
                print(f"{split} split:")
                print(f"  - Users: {len(data.get('user_id', []))}")
                print(f"  - Items: {len(data.get('item_id', []))}")
                print(f"  - Sequences: {data.get('item_seq', torch.tensor([])).shape}")
                print(f"  - Sequence lengths: {data.get('item_seq_len', torch.tensor([])).shape}")
        
        # Test item features
        if reader.use_item_features:
            test_item_ids = [0, 1, 2]
            features = reader.get_item_features(test_item_ids)
            print(f"Item features for items {test_item_ids}: {len(features)} found")
            
            tokens = reader.get_item_tokens(test_item_ids)
            print(f"Item tokens for items {test_item_ids}: {len(tokens)} found")
        
        print("✅ LETTERReader test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LETTERReader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LETTER integration")
    parser.add_argument("--dataset", type=str, default="TestBeauty", help="Test dataset name")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files after testing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LETTER Integration Test")
    print("=" * 60)
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test data
        dataset_dir = create_test_data(temp_dir, args.dataset)
        
        # Test LETTERReader
        success = test_letter_reader(dataset_dir, args.dataset)
        
        if success:
            print("\n🎉 All tests passed! LETTER integration is working correctly.")
        else:
            print("\n💥 Tests failed! Please check the error messages above.")
            sys.exit(1)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
