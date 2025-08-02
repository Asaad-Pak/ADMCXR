import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import json
import queue
import copy
import random
from tqdm import tqdm
import argparse
import os

# Path for the processed IDs file
# PROCESSED_IDS_PATH = r'F:\Asaad_Output\Preprocessing\ITM\Validation_Data\simple_output\processed_ids.txt'  # for validation simple output
# PROCESSED_IDS_PATH = r'F:\Asaad_Output\Preprocessing\ITM\Validation_Data\pretty_output\processed_ids.txt'  # for validation pretty output

# PROCESSED_IDS_PATH = r'F:\Asaad_Output\Preprocessing\ITM\Test_Data\simple_output\processed_ids.txt' # for test simple output
# PROCESSED_IDS_PATH = r'F:\Asaad_Output\Preprocessing\ITM\Test_Data\pretty_output\processed_ids.txt' # for test pretty output

# PROCESSED_IDS_PATH = r'F:\Asaad_Output\Preprocessing\ITM\Training_Data\pretty_output\processed_ids.txt' # for training pretty output
PROCESSED_IDS_PATH = r'F:\Asaad_Output\Preprocessing\ITM\TR_Data_150K\pretty_output\processed_ids.txt' # for training 150k pretty output

def initialize_processed_ids_file(file_path): #new
    """Ensure the processed IDs file exists, create an empty one if not."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # Create an empty file

def load_processed_ids(file_path): #new
    """Load processed IDs from the log file."""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def log_processed_id(file_path, dicom_ids):  #new
    """Log a processed ID into the log file."""
    with open(file_path, 'a') as f:
        for dicom_id in dicom_ids: #jk
            f.write(f"{dicom_id}\n")

def load_existing_json(save_path): #new
    """Load existing content from the JSON file."""
    if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
        return []
    with open(save_path, 'r') as f:
        content = f.read().strip()
        if content.endswith(',\n]'):  # Handle potential trailing comma issue
            content = content[:-3] + ']'
        return json.loads(content)

def append_to_json(save_path, data): #new
    """Append new data to the JSON file."""
    existing_data = load_existing_json(save_path)
    with open(save_path, 'w') as f:
        # json.dump(existing_data + data, f) # for json rough printing
        json.dump(existing_data + data, f,indent=4) #for json pretty printing

def finalize_json(save_path): #new
    """Ensure the JSON file is valid by removing the last comma and closing the JSON array."""
    with open(save_path, 'r+') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()

        if size == 0:  # If file is empty
            f.write('[]')  # Write an empty JSON array
        else:
            f.seek(0)
            content = f.read().strip()
            if content == '[':  # If file only contains '['
                f.seek(0)
                f.write('[]')  # Write an empty JSON array
            else:
                # Ensure proper truncation and closure
                f.seek(0, os.SEEK_END)
                pos = f.tell() - 1
                while pos > 0 and f.read(1) != ',':
                    pos -= 1
                    f.seek(pos, os.SEEK_SET)
                if pos > 0:
                    f.seek(pos, os.SEEK_SET)
                    f.truncate()  # Remove the trailing comma
                f.write('\n]')  # Close the JSON array

def main(args): 
    train_impressions = pd.read_json(args.train_path) 
    train_impressions = train_impressions.set_index('dicom_id')
    dicom_ids = set(train_impressions.index)
    train_impressions_chexbert = pd.read_csv(args.train_chexbert_path)

    # Ensure processed_ids.txt exists and load previously processed IDs
    initialize_processed_ids_file(PROCESSED_IDS_PATH) #new
    processed_ids = load_processed_ids(PROCESSED_IDS_PATH) #new

    dataset = defaultdict(lambda: queue.Queue())

    try: #new try excep block wasn't there before it is just to handle interruption
        for idx, row in train_impressions_chexbert.iterrows():
            dicom_id, study_id, subject_id = row['dicom_id'], row['study_id'], row['subject_id']
            
            # Skip if dicom_id is already processed (whole if statment is #new)
            if dicom_id in processed_ids:
                print(f"Skipping already processed ID: {dicom_id}")
                continue

            # Add the print statement here to monitor processing #new
            print("Processing:", dicom_id) 

            if dicom_id not in dicom_ids:
                continue
            # label = tuple(row[4:])
            label = tuple([0 if pd.isna(val) else int(val) for val in row[4:]])
            assert len(label) == 14, 'length of the chexbert label detected is not 14'
            dataset[label].put({'dicom_id': dicom_id, 'study_id': study_id, 'sentence': row['Report Impression'], 'image': train_impressions['image'][dicom_id], 'label': None})

        total_keys = list(dataset.keys())
        print("Keys: ", total_keys)
        print(f"Number of unique keys in dataset: {len(dataset.keys())}")

        total_processed = sum(dataset[key].qsize() for key in dataset.keys())
        print(f"Total rows processed: {total_processed}")
        total_keys_freq = np.array([dataset[k].qsize() for k in total_keys], dtype = np.float64)
        total_keys_freq /= np.sum(total_keys_freq)

        train_files = []
        edits = np.zeros(14)
        for key in tqdm(dataset.keys()):
            dicom_ids_to_log = set() #jk
            print("Key being examined: ", key)

            for i in range(dataset[key].qsize()):
                if i % 1000 == 0:
                    print(i, dataset[key].qsize())
                el = dataset[key].get()
                dataset[key].put(el)

                positive = copy.deepcopy(el)
                positive['label'] = 'positive'
                train_files.append(positive)

                hard_negative = copy.deepcopy(el)
                hard_negative['label'] = 'negative'
                hard_negative_keys = []
                hard_negative_freq = []
                for edit_distance in range(1, 14):
                    for cand in dataset.keys():
                        if cand == key:
                            continue
                        v = np.array(key) - np.array(cand)
                        if np.sum(np.abs(v)) <= edit_distance:
                            hard_negative_keys.append(cand)
                            hard_negative_freq.append(dataset[cand].qsize())
                    # if len(hard_negative_keys) == 0: #new just for debugging
                    #     print(f"No candidates found with edit distance <= {edit_distance} for key {key}")
                    if len(hard_negative_keys) > 0:
                        edits[edit_distance] += 1
                        break
                
                if len(hard_negative_keys) > 0:
                    hard_negative_freq = np.array(hard_negative_freq, dtype = np.float64)
                    hard_negative_freq /= np.sum(hard_negative_freq)
                    
                    x = np.random.choice(len(hard_negative_keys), 1, p=hard_negative_freq)[0]
                    hard_negative_keys = hard_negative_keys[x]
                    hard_negative_cand = dataset[hard_negative_keys].get()
                    hard_negative['sentence'] = hard_negative_cand['sentence']
                    hard_negative['hard_negative_dicom_id'] = hard_negative_cand['dicom_id']
                    hard_negative['edit_distance'] = edit_distance
                    dataset[hard_negative_keys].put(hard_negative_cand)
                    train_files.append(hard_negative)
                else:
                    print(f"No hard negatives found for key {key}, skipping.")

                negative = copy.deepcopy(el)
                negative['label'] = 'negative'
                while True:
                    negative_key = total_keys[np.random.choice(np.arange(len(total_keys)), p=total_keys_freq)]
                    print("Negative_key selected as candidate ", negative_key)
                    # print("Current key examined here in simple negative block: ", key)
                    if negative_key != key:
                        print("Simple negative found") #new
                        break

                # print("DRACULA")
                negative_cand = dataset[negative_key].get()
                dataset[negative_key].put(negative_cand)
                negative['sentence'] = negative_cand['sentence']
                negative['negative_sentence_dicom_id'] = negative_cand['dicom_id']
                # Ensure edit_distance is JSON-serializable
                negative['edit_distance'] = int(np.sum(np.abs(np.array(negative_key) - np.array(key))))
                # negative['edit_distance'] = float(np.sum(np.abs(np.array(negative_key) - np.array(key))))
                # negative['edit_distance'] = np.sum(np.abs(np.array(negative_key) - np.array(key)))
                train_files.append(negative)

                dicom_ids_to_log.add(el['dicom_id'])

                print("------------------------------")
            # print("Training Files: ",train_files)
            # Append new data to JSON #new
            print("*****************************")
            append_to_json(args.save_path, train_files)
            train_files.clear()

            # Log the processed ID only after completing all computations for the dicom_id
            # log_processed_id(PROCESSED_IDS_PATH, el['dicom_id']) #new
            log_processed_id(PROCESSED_IDS_PATH, dicom_ids_to_log) #jk

            # try:
            #     queue_size = dataset[key].qsize()  # Get the initial size of the queue
            #     print(f"Processing {queue_size} items for key {key}...")
                
            #     for _ in range(queue_size):
            #         dicom_id_entry = dataset[key].get()
            #         if dicom_id_entry:  # Check if the dequeued item is valid
            #             print(f"Logging processed ID: {dicom_id_entry['dicom_id']}")
            #             log_processed_id(PROCESSED_IDS_PATH, dicom_id_entry['dicom_id'])
            #         else:
            #             print(f"Invalid entry encountered for key {key}")
            # except Exception as e:
            #     print(f"Error while processing IDs for key {key}: {e}")


    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Finalizing JSON...")
        finalize_json(args.save_path)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path to the train file (in csv format) used to create the trainining set for image-text matching fine-tuning')
    parser.add_argument('--train_chexbert_path', help='path to the chexbert labels (in csv format) for the train file')
    parser.add_argument('--save_path', help='path to dump the output' )
    args = parser.parse_args()
    main(args)
