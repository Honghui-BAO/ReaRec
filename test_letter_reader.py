#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Simple test for LETTER reader integration.
"""

import os
import sys
import json
import tempfile
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_letter_data(output_dir, dataset_name="TestBeauty"):
    """
    Create test data in LETTER format.
    """
    print(f"Creating test LETTER data for {dataset_name}...")
    
    # Create test interactions (user_id -> list of item_ids)
    test_interactions = {
        "0": [0, 1, 2, 3, 4, 5],  # User 0 has 6 interactions
        "1": [1, 2, 3, 4, 5, 6, 7],  # User 1 has 7 interactions
        "2": [2, 3, 4, 5, 6, 7, 8, 9],  # User 2 has 8 interactions
    }
    
    # Create test item indices (item_id -> list of tokens in LETTER format)
    test_indices = {
        "0": ["<a_123>", "<b_45>", "<c_67>", "<d_89>"],
        "1": ["<a_234>", "<b_56>", "<c_78>", "<d_90>"],
        "2": ["<a_345>", "<b_67>", "<c_89>", "<d_01>"],
        "3": ["<a_456>", "<b_78>", "<c_90>", "<d_12>"],
        "4": ["<a_567>", "<b_89>", "<c_01>", "<d_23>"],
        "5": ["<a_678>", "<b_90>", "<c_12>", "<d_34>"],
        "6": ["<a_789>", "<b_01>", "<c_23>", "<d_45>"],
        "7": ["<a_890>", "<b_12>", "<c_34>", "<d_56>"],
        "8": ["<a_901>", "<b_23>", "<c_45>", "<d_67>"],
        "9": ["<a_012>", "<b_34>", "<c_56>", "<d_78>"],
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
    
    print(f"Test LETTER data created in {dataset_dir}")
    return dataset_dir

def test_letter_reader_basic(dataset_dir, dataset_name):
    """
    Test basic LETTER reader functionality without dependencies.
    """
    print(f"Testing LETTER reader basic functionality with {dataset_name}...")
    
    # Test file loading
    inter_file = os.path.join(dataset_dir, f"{dataset_name}.inter.json")
    index_file = os.path.join(dataset_dir, f"{dataset_name}.index.json")
    item_file = os.path.join(dataset_dir, f"{dataset_name}.item.json")
    
    try:
        # Load interactions
        with open(inter_file, 'r') as f:
            inters = json.load(f)
        print(f"✅ Loaded {len(inters)} user interactions")
        
        # Load indices
        with open(index_file, 'r') as f:
            indices = json.load(f)
        print(f"✅ Loaded {len(indices)} item tokenizations")
        
        # Load item features
        with open(item_file, 'r') as f:
            items = json.load(f)
        print(f"✅ Loaded {len(items)} item features")
        
        # Test data structure
        print("\nData structure validation:")
        
        # Check interactions format
        for uid, item_list in list(inters.items())[:2]:  # Check first 2 users
            print(f"  User {uid}: {len(item_list)} items - {item_list}")
        
        # Check tokenization format
        for item_id, tokens in list(indices.items())[:2]:  # Check first 2 items
            print(f"  Item {item_id}: {tokens}")
            if len(tokens) == 4 and all(token.startswith('<') and token.endswith('>') for token in tokens):
                print(f"    ✅ Valid LETTER tokenization format")
            else:
                print(f"    ❌ Invalid tokenization format")
        
        # Check item features format
        for item_id, features in list(items.items())[:2]:  # Check first 2 items
            print(f"  Item {item_id}: {features}")
            if 'title' in features and 'description' in features:
                print(f"    ✅ Valid item features format")
            else:
                print(f"    ❌ Invalid item features format")
        
        print("\n🎉 Basic LETTER reader test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic LETTER reader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test LETTER reader integration")
    parser.add_argument("--dataset", type=str, default="TestBeauty", help="Test dataset name")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LETTER Reader Integration Test")
    print("=" * 60)
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test data
        dataset_dir = create_test_letter_data(temp_dir, args.dataset)
        
        # Test basic functionality
        success = test_letter_reader_basic(dataset_dir, args.dataset)
        
        if success:
            print("\n🎉 All basic tests passed! LETTER reader integration is working correctly.")
            print("\nNext steps:")
            print("1. Activate the conda environment: conda activate rrec")
            print("2. Run the full test: python test_letter_integration.py")
            print("3. Use the integration: python src/main.py --use_letter_reader 1")
        else:
            print("\n💥 Basic tests failed! Please check the error messages above.")
            sys.exit(1)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
