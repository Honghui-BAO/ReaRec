#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Simple test to verify the LETTER integration files are properly created.
"""

import os
import json

def test_file_creation():
    """Test that all required files are created."""
    print("Testing LETTER integration file creation...")
    
    base_dir = "/Users/honghuibao/Desktop/ReaRec"
    
    # Check if files exist
    files_to_check = [
        "src/helpers/LETTERReader.py",
        "src/helpers/preprocess_letter_data.py", 
        "run_letter_example.sh",
        "test_letter_integration.py",
        "LETTER_INTEGRATION.md"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_file_content():
    """Test that files contain expected content."""
    print("\nTesting file content...")
    
    base_dir = "/Users/honghuibao/Desktop/ReaRec"
    
    # Test LETTERReader.py
    reader_file = os.path.join(base_dir, "src/helpers/LETTERReader.py")
    if os.path.exists(reader_file):
        with open(reader_file, 'r') as f:
            content = f.read()
            if "class LETTERReader" in content and "rating_threshold" in content:
                print("✅ LETTERReader.py contains expected content")
            else:
                print("❌ LETTERReader.py missing expected content")
    
    # Test main.py modifications
    main_file = os.path.join(base_dir, "src/main.py")
    if os.path.exists(main_file):
        with open(main_file, 'r') as f:
            content = f.read()
            if "use_letter_reader" in content and "LETTERReader" in content:
                print("✅ main.py contains LETTER integration")
            else:
                print("❌ main.py missing LETTER integration")

def test_data_structure():
    """Test that we can create test data structure."""
    print("\nTesting data structure creation...")
    
    # Create test data structure
    test_data = {
        "interactions": {
            "0": [0, 1, 2, 3, 4],
            "1": [1, 2, 3, 4, 5, 6]
        },
        "indices": {
            "0": ["<a_0>", "<b_1>"],
            "1": ["<a_1>", "<b_2>"]
        },
        "items": {
            "0": {"title": "Test Item 0", "description": "Test description"},
            "1": {"title": "Test Item 1", "description": "Test description"}
        }
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(test_data, indent=2)
        parsed_data = json.loads(json_str)
        if parsed_data == test_data:
            print("✅ JSON data structure works correctly")
        else:
            print("❌ JSON data structure parsing failed")
    except Exception as e:
        print(f"❌ JSON data structure test failed: {e}")

def main():
    print("=" * 60)
    print("LETTER Integration - Simple Test")
    print("=" * 60)
    
    # Run tests
    file_test = test_file_creation()
    test_file_content()
    test_data_structure()
    
    print("\n" + "=" * 60)
    if file_test:
        print("🎉 Basic integration test passed!")
        print("\nNext steps:")
        print("1. Activate the conda environment: conda activate rrec")
        print("2. Run the full test: python test_letter_integration.py")
        print("3. Use the integration: python src/main.py --use_letter_reader 1")
    else:
        print("💥 Some files are missing. Please check the file creation.")
    print("=" * 60)

if __name__ == "__main__":
    main()
