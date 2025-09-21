# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import pandas as pd
import torch
import json
from torch.nn.functional import pad
from collections import defaultdict

from utils import utils
from utils.constants import *


class LETTERReader(object):
    """
    LETTER data reader class: Loads and processes LETTER datasets with leave-one-out splitting
    and tokenized item representations.
    """

    @staticmethod
    def parse_data_args(parser):
        """
        Parses command-line arguments related to LETTER data loading.
        :param parser: argparse.ArgumentParser, argument parser instance.
        :return: argparse.ArgumentParser, updated argument parser.
        """
        parser.add_argument(
            "--path",
            type=str,
            default="../datasets/processed",
            help="Input data dir.",
        )
        parser.add_argument(
            "--dataset", type=str, default="Beauty", help="Choose a dataset."
        )
        parser.add_argument(
            "--use_item_features", 
            action="store_true", 
            help="Whether to load item features (title, description)"
        )
        return parser

    def __init__(self, args):
        """
        Initializes LETTERReader, loads, and processes data.
        :param args: argparse.Namespace, contains dataset path, name, etc.
        """
        self.prefix = args.path
        self.dataset = args.dataset
        self.use_item_features = args.use_item_features
        self._read_data()

    def _read_data(self):
        """
        Reads and preprocesses the LETTER dataset.
        """
        logging.info(
            'Reading LETTER data from "{}", dataset = "{}" '.format(self.prefix, self.dataset)
        )
        
        # Load interaction data
        inter_file = os.path.join(self.prefix, self.dataset, f"{self.dataset}.inter.json")
        with open(inter_file, 'r') as f:
            self.inters = json.load(f)
        
        # Load item index mapping (tokenization)
        index_file = os.path.join(self.prefix, self.dataset, f"{self.dataset}.index.json")
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        # Load item features if requested
        if self.use_item_features:
            item_file = os.path.join(self.prefix, self.dataset, f"{self.dataset}.item.json")
            with open(item_file, 'r') as f:
                self.item_features = json.load(f)
        
        # Process data with leave-one-out splitting (LETTER's approach)
        self._process_letter_data()
        
        logging.info("Finish reading LETTER data.")

    def _process_letter_data(self):
        """
        Process LETTER data with leave-one-out splitting.
        """
        logging.info("Processing LETTER data with leave-one-out splitting...")
        
        # Convert string keys to integers for interactions
        processed_inters = {}
        for uid, items in self.inters.items():
            processed_inters[int(uid)] = [int(item) for item in items]
        
        # Calculate dataset statistics
        all_items = set()
        for items in processed_inters.values():
            all_items.update(items)
        
        self.n_users = len(processed_inters) + 1  # including [PAD]
        self.n_items = max(all_items) + 2  # including [PAD]
        
        logging.info(
            '"# user": {}, "# item": {}, "# entry": {}'.format(
                self.n_users - 1, self.n_items - 1, sum(len(items) for items in processed_inters.values())
            )
        )
        
        # Convert to ReaRec format with leave-one-out splitting
        self._convert_to_rearec_format(processed_inters)

    def _convert_to_rearec_format(self, processed_inters):
        """
        Convert LETTER data format to ReaRec format using leave-one-out splitting.
        """
        train_data = []
        valid_data = []
        test_data = []
        
        for uid, items in processed_inters.items():
            if len(items) < 3:  # Skip users with too few interactions
                continue
                
            # Leave-one-out splitting: last item for test, second last for validation
            train_items = items[:-2]
            valid_item = items[-2]
            test_item = items[-1]
            
            # Create training samples (all but last two items)
            for i in range(1, len(train_items)):
                train_data.append({
                    USER_ID: uid,
                    ITEM_ID: train_items[i],
                    ITEM_SEQ: train_items[:i]
                })
            
            # Create validation sample
            valid_data.append({
                USER_ID: uid,
                ITEM_ID: valid_item,
                ITEM_SEQ: items[:-2]
            })
            
            # Create test sample
            test_data.append({
                USER_ID: uid,
                ITEM_ID: test_item,
                ITEM_SEQ: items[:-1]
            })
        
        # Convert to DataFrame format
        self.data_dict = {}
        for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
            if split_data:
                df = pd.DataFrame(split_data)
                # Truncate sequences to max length
                df[ITEM_SEQ] = df[ITEM_SEQ].apply(lambda x: x[-MAX_ITEM_SEQ_LEN:])
                df[ITEM_SEQ_LEN] = df[ITEM_SEQ].apply(len)
                
                # Convert to tensors
                self.data_dict[split_name] = self._dataframe_to_tensors(df)
            else:
                self.data_dict[split_name] = {}
        
        logging.info(f"size of train: {len(train_data)}")
        logging.info(f"size of valid: {len(valid_data)}")
        logging.info(f"size of test: {len(test_data)}")

    def _dataframe_to_tensors(self, df):
        """
        Convert DataFrame to tensor format compatible with ReaRec.
        """
        data_dict = {}
        
        # Convert item sequences to tensors
        item_seq = [torch.from_numpy(np.array(x)).long() for x in df[ITEM_SEQ].values]
        
        # Left Padding
        left_padded_seqs = [
            pad(seq, (MAX_ITEM_SEQ_LEN - len(seq), 0), value=self.n_items - 1)
            for seq in item_seq
        ]
        data_dict[ITEM_SEQ] = torch.stack(left_padded_seqs)
        
        # Other fields
        data_dict[USER_ID] = torch.from_numpy(df[USER_ID].values).long()
        data_dict[ITEM_ID] = torch.from_numpy(df[ITEM_ID].values).long()
        data_dict[ITEM_SEQ_LEN] = torch.from_numpy(df[ITEM_SEQ_LEN].values).long()
        
        return data_dict

    def get_item_features(self, item_ids):
        """
        Get item features for given item IDs.
        :param item_ids: list of item IDs
        :return: dict of item features
        """
        if not self.use_item_features:
            return None
        
        features = {}
        for item_id in item_ids:
            if str(item_id) in self.item_features:
                features[item_id] = self.item_features[str(item_id)]
        
        return features

    def get_item_tokens(self, item_ids):
        """
        Get tokenized representation for given item IDs.
        :param item_ids: list of item IDs
        :return: dict of item tokens
        """
        tokens = {}
        for item_id in item_ids:
            if str(item_id) in self.indices:
                tokens[item_id] = self.indices[str(item_id)]
        
        return tokens
