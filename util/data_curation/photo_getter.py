import os
import requests
import concurrent.futures

DATASET_DIR = "/home/richard/Desktop/flutter/Mushroom_Classifier/assets/data/mushroom_photos2"
mushroom_urls = []

def make_directory(mush_dir):
    try:
        os.makedirs(mush_dir)  # Create directory if it doesn't exist
    except Exception as e:
        pass

def get_photos(data):
    id, name, url = data
    mush_dir = os.path.join(DATASET_DIR, name)
    ext = url.split('.')[-1]
    make_directory(mush_dir)
    
    try:
        response = requests.get(url, timeout=10)  # Set a timeout for requests
        if response.status_code == 200:
            photo_file = os.path.join(mush_dir, f"{id}.{ext}")
            with open(photo_file, "xb") as f:  # Use "xb" to avoid overwriting existing files
                f.write(response.content)
                # print(f"{photo_file} written.") 
        else:
            print(f"Failed to download {url} with status code {response.status_code}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def load_mushroom_urls(file_path):
    urls = []
    with open(file_path, 'r') as f:
        f.readline()
        for line in f.readlines():
            line_split = line.strip().split(',')
            id = line_split[0]
            name = line_split[1]
            url = line_split[5]
            urls.append((id, name, url))
    return urls

# Load URLs
response_file = "/home/richard/Desktop/flutter/Mushroom_Classifier/assets/data/inat/response2.csv"
mushroom_urls = load_mushroom_urls(response_file)
print(f"Total URLs: {len(mushroom_urls)}")

# Multithreading for downloading
max_threads = 10  # Adjust based on your system and network capabilities

with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    futures = [executor.submit(get_photos, data) for data in mushroom_urls]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # Raises exception if the thread failed
        except Exception as e:
            print(f"Thread failed: {e}")
