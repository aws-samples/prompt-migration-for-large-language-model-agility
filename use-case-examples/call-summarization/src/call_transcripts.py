# Download from s3 everything in a folder
import os
import boto3
from pathlib import Path
import json
import pandas
import random


# Insert random character in a random location in a string
def insert_random_char(string, char):
    random_index = random.randint(0, len(string) - 1)
    return string[:random_index] + char + string[random_index:]

# delete random index from a string
def delete_random_char(string):
    random_index = random.randint(0, len(string) - 1)
    return string[:random_index] + string[random_index + 1:]

# append string value to end of a txt file - add it in a new line
def append_txt(filename, string):
    with open(filename, "a") as f:
        f.write(string + "\n")


# get all files inside a folder in local
def get_files(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files

def download_folder(bucket, prefix, local_path):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=prefix):
        target = os.path.join(local_path, obj.key)
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
    return local_path

class Items:
    """ Class to hold information about a single speech segment """
    def __init__(self):
        self.segmentStartTime = 0.0
        self.segmentEndTime = 0.0
        self.segmentSpeaker = ""
        self.segmentText = ""
        self.type = ""
        self.confidence = 0.0

class SpeechSegment:
    """ Class to hold information about a single speech segment """
    def __init__(self):
        self.segmentStartTime = 0.0
        self.segmentEndTime = 0.0
        self.speakerLabel = ""
        self.items = []

        # Not in original version, so may not exist in legacy files
        self.segmentIVR = False

def bool_expr(items_data, item_idx, segment): 
    return item_idx < len(items_data) and \
            ("start_time" not in items_data[item_idx].keys() or \
             float(items_data[item_idx]['start_time']) < segment.segmentEndTime)

def get_transcript_str(filename): 
    json_filepath = Path(f"data/{filename}")
    json_data = json.load(open(json_filepath.absolute(), "r", encoding="utf-8"))
    speech_segments = []
    item_idx = 0

    # check if transcript is empty
    if not json_data["results"]["items"]:
        return
    for next_segment in json_data["results"]['speaker_labels']['segments']:
        new_segment = SpeechSegment()

        # Standard segment data
        new_segment.segmentStartTime = float(next_segment["start_time"])
        new_segment.segmentEndTime = float(next_segment["end_time"])
        new_segment.speakerLabel = next_segment['speaker_label']
        while bool_expr(json_data['results']['items'], item_idx, new_segment):
            item_segment = Items()
            item = json_data['results']['items'][item_idx]
            if "start_time" in item.keys(): 
                start_time = item['start_time']
                end_time = item['end_time']
            item_segment.segmentStartTime = start_time
            item_segment.segmentEndTime = end_time
            item_segment.type = item['type']
            if 'speaker_label' in item.keys():
                item_segment.segmentSpeaker = item['speaker_label']
            for text in item['alternatives']: 
                item_segment.segmentText += text['content'] + " "
            new_segment.items += [item_segment]
            item_idx += 1    

        # Add what we have to the full list
        speech_segments.append(new_segment)
        
    transcript_str = ""
    for segment in speech_segments: 
        transcript_str += segment.speakerLabel + ": "
        for item in segment.items: 
            transcript_str += item.segmentText
        transcript_str += '\n'
    
    # Split filename by / and remove .json at the end
    filename = filename.split('/')[-1]
    filename = filename[:-5]


    # file upload
    with open(f"data/transcripts/{filename}.txt", "w") as file:
        file.write(transcript_str)
    return transcript_str

# metadata prompts
metadata_prompts = {
    "summary": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>What is a summary of the transcript?</question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere is the call summary: ",
    "topic": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>What is the topic of the call? For example, iphone issue, billing issue, cancellation. Only reply with the topic, nothing more.</question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere is the topic: ",
    "product": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>What product did the customer call about? For example, internet, broadband, mobile phone, mobile plans. Only reply with the product, nothing more.</question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere is the product: ",
    "root_cause": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>What is the main reason/cause of the call?</question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere is the root cause: ",
    "issue_resolved": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>Did the agent resolve the customer's questions? Only reply with yes or no, nothing more. </question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere is the answer to whether the issue was resolved or not: ", 
    "callback": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>Was this a callback? (yes or no) Only reply with yes or no, nothing more.</question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere is the answer to whether the call was a callback or not: ", 
    "next_steps": "\n\nHuman: Answer the questions below, defined in <question></question> based on the transcript defined in <transcript></transcript>. If you cannot answer the question, reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the answer. Do not use XML tags in the answer.\n\n<question>What actions did the Agent take? </question>\n\n<transcript>\n{transcript}\n</transcript>\n\nAssistant:\nHere are the next steps: "
}

def get_call_metadata(call_transcript, prompt_type): 
    prompt = prompt_type.replace("{transcript}", call_transcript)
    generated_text = call_bedrock(parameters, prompt)
    return generated_text
