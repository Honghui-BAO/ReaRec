# LETTER Data Integration for ReaRec

This document describes how to use the integrated LETTER data reading functionality in ReaRec.

## Overview

The integration allows ReaRec to work with LETTER's data format, which includes:
- **Absolute timestamp splitting**: Data is split based on actual timestamps rather than leave-one-out
- **Rating filtering**: Only interactions with rating > 3 are considered positive samples
- **JSON format**: Uses `.inter.json`, `.index.json`, and `.item.json` files
- **Item features**: Supports item titles and descriptions

## Key Differences

| Aspect | ReaRec (Original) | LETTER Integration |
|--------|------------------|-------------------|
| Data Format | CSV files (`.remap.csv`) | JSON files (`.inter.json`, etc.) |
| Data Split | Leave-one-out | Leave-one-out (same as LETTER) |
| Rating Filter | Rating > 3 as positive | Not applicable (LETTER data already filtered) |
| Item Features | Not supported | Title, description support |
| Item Tokenization | Not supported | 4-token representation (`<a_xxx>`, `<b_xxx>`, `<c_xxx>`, `<d_xxx>`) |

## Files Added

1. **`src/helpers/LETTERReader.py`**: Main data reader for LETTER format
2. **`src/helpers/preprocess_letter_data.py`**: Script to convert raw data to LETTER format
3. **`run_letter_example.sh`**: Example script to run ReaRec with LETTER reader
4. **`LETTER_INTEGRATION.md`**: This documentation

## Usage

### 1. Data Preparation

First, convert your raw data to LETTER format:

```bash
python src/helpers/preprocess_letter_data.py \
    --dataset Beauty \
    --input_path /path/to/raw/data \
    --output_path /path/to/processed/data \
    --rating_threshold 3.0 \
    --min_interactions 5 \
    --min_items 5
```

This will create the following files in the output directory:
- `{dataset}.inter.json`: User-item interactions (leave-one-out format)
- `{dataset}.index.json`: Item tokenization indices (4-token representation)
- `{dataset}.item.json`: Item features (title, description)

### 2. Running ReaRec with LETTER Reader

Use the `--use_letter_reader 1` flag to enable LETTER data reading:

```bash
python src/main.py \
    --model_name PRL \
    --dataset Beauty \
    --use_letter_reader 1 \
    --use_item_features 1 \
    --gpu 0 \
    --train 1
```

### 3. Key Parameters

- `--use_letter_reader 1`: Enable LETTER data reader
- `--use_item_features 1`: Load item features (title, description)

## Data Format

### Input Format (LETTER)

The LETTER reader expects the following JSON files:

**`{dataset}.inter.json`**:
```json
{
    "0": [0, 1, 2, 3, 4],
    "1": [5, 6, 7, 8, 9, 3, 10],
    "2": [3, 11, 12, 13]
}
```

**`{dataset}.index.json`**:
```json
{
    "0": ["<a_123>", "<b_45>", "<c_67>", "<d_89>"],
    "1": ["<a_234>", "<b_56>", "<c_78>", "<d_90>"]
}
```

**`{dataset}.item.json`**:
```json
{
    "0": {
        "title": "Item Title",
        "description": "Item Description"
    }
}
```

### Output Format (ReaRec Compatible)

The LETTER reader converts the data to ReaRec's expected format:
- User sequences with proper padding
- Leave-one-out splitting (last item for test, second last for validation)
- Tensor format compatible with existing models

## Integration Details

### Data Processing Flow

1. **Load JSON files**: Read interactions, indices, and item features
2. **Process interactions**: Convert string keys to integers
3. **Apply leave-one-out splitting**: Last item for test, second last for validation
4. **Convert to ReaRec format**: Transform to tensor format
5. **Create tensors**: Convert to PyTorch tensors with proper padding

### Compatibility

The LETTER reader is fully compatible with existing ReaRec models:
- **PRL (Progressive Reasoning Learning)**
- **ERL (Ensemble Reasoning Learning)**
- All other sequential recommendation models

### Performance Considerations

- The LETTER reader loads all data into memory for processing
- For large datasets, consider using the preprocessing script to filter data first
- Item features are optional and can be disabled to save memory

## Example Workflow

1. **Prepare raw data** in CSV format with columns: `user_id`, `item_id`, `rating`, `timestamp`
2. **Run preprocessing** to convert to LETTER format
3. **Train model** using LETTER reader
4. **Evaluate** using standard ReaRec evaluation metrics

## Troubleshooting

### Common Issues

1. **File not found**: Ensure all required JSON files exist in the dataset directory
2. **Memory issues**: Reduce dataset size or disable item features
3. **Format errors**: Check JSON file format and ensure proper encoding

### Debug Mode

Enable verbose logging to debug data loading:

```bash
python src/main.py --use_letter_reader 1 --verbose 10
```

## Future Enhancements

Potential improvements for the LETTER integration:
- Support for more complex item features
- Batch processing for large datasets
- Integration with LETTER's tokenization system
- Support for multi-modal features

## References

- [ReaRec Paper](https://arxiv.org/abs/2503.22675)
- [LETTER Paper](https://arxiv.org/abs/2405.07314)
- Original ReaRec implementation
- LETTER project repository
