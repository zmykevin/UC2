import os, requests, uuid, json
import csv
import sys
from tqdm import tqdm
from shutil import copyfile

csv.field_size_limit(sys.maxsize)

key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))

subscription_key = os.environ[key_var_name]
endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

# If you encounter any issues with the base_url or path, make sure
# that you are using the latest endpoint: https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
path = '/translate?api-version=3.0'
constructed_url = endpoint + path

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

def msft_translate(lines, langs, batch_size=10):
    global constructed_url, headers
    if 'zh' in langs:
        langs = langs.replace('zh', 'zh-Hans')
    params = "&" + "&".join(["to={}".format(lang) for lang in langs.split(',')]) + "&includeAlignment=true&includeSentenceLength=true"
    request_url = constructed_url + params
    # Body limitations:
    #    The array can have at most 100 elements.
    #    The entire text included in the request cannot exceed 5,000 characters including spaces.
    out_translations = []

    #for idx in tqdm(range(0, len(lines), batch_size)):
    for idx in range(0, len(lines), batch_size):
        body = [{'text': line} for line in lines[idx:idx+batch_size]]

        request = requests.post(request_url, headers=headers, json=body)
        response = request.json()

        out_translations += response

    return out_translations

# in_file = ''
# out_file = ''
# tgt_langs = 'en'
# batch_size = 10
split = "train"
in_file = "/home/mingyang/multilingual_vl/data/azure_mount/UNITER_DATA/CC/annotations/{split}_imageId2Ann.tsv".format(split=split)
tgt_langs = 'fr'
backup_file = "/home/mingyang/multilingual_vl/data/azure_mount/UNITER_DATA/CC/annotations/{split}_imageId2Ann_{tgt_langs}_backup.tsv".format(split=split, tgt_langs=tgt_langs)
out_file = "/home/mingyang/multilingual_vl/data/azure_mount/UNITER_DATA/CC/annotations/{split}_imageId2Ann_{tgt_langs}.tsv".format(split=split, tgt_langs=tgt_langs)
alignment_out_file = "/home/mingyang/multilingual_vl/data/azure_mount/UNITER_DATA/CC/annotations/{split}_en_{tgt_langs}_word_alignemnt.tsv".format(split=split, tgt_langs=tgt_langs)
line_batch_size=100
#batch_size = 100
save_step = 10000
#with open(in_file, "r") as tsv_src:
tsv_src = open(in_file)

#get the tsv_tgt_data
tsv_tgt_data = []
src_captions = []
#src_tgt_alignments = []

if os.path.isfile(backup_file):
    #load the tsv_tgt_data
    tsv_tgt_backup = open(backup_file)
    for tgt_line in tsv_tgt_backup:
        img_id, img_url, tgt_caption, success = tgt_line.strip().split('\t')
        tsv_tgt_data.append([img_id, img_url, tgt_caption, success])
#set up the resume point
resume_step = len(tsv_tgt_data)
print("resume_step is: {}".format(resume_step))
for i, src_line in enumerate(tsv_src):
    img_id, img_url, src_caption, success = src_line.strip().split('\t')
    src_captions.append(src_caption)
    #split the fields
    if i >= resume_step:
        tsv_tgt_data.append([img_id, img_url, "", success])
        #src_tgt_alignments.append([img_id, ""])

#print(len(tsv_tgt_data) == len(src_captions))
for i in tqdm(range(resume_step, len(src_captions), line_batch_size)):
    if i + line_batch_size < len(src_captions):
        current_src_batch = src_captions[i:i+line_batch_size]
    else:
        current_src_batch = src_captions[i:]
    
    #get the translations
    batch_size = len(current_src_batch)
    current_translation_batch = msft_translate(current_src_batch, langs=tgt_langs, batch_size=batch_size)
    #update tsv_tgt_data
    for j in range(len(current_translation_batch)):
        if current_translation_batch[j]['translations'][0].get('text', None):
            translated_text = current_translation_batch[j]['translations'][0]['text']
            if len(translated_text) > 10000:
                tsv_tgt_data[i+j][3] = "fail"
            else:
                tsv_tgt_data[i+j][2] = translated_text
        #     print(current_translation_batch[j])  
        #     raise Exception("something wrong happens to the translation: {}".format(current_translation_batch[j]))
        # finally: 
        #     print(current_translation_batch[j])    
             #retried_translation = msft_translate()
             # if current_translation_batch[j]['translations'][0].get('alignment', None):
            #     src_tgt_alignments[i+j][1] = current_translation_batch[j]['translations'][0]['alignment']['proj']
    if (i+1) % save_step == 1:
        saved_tsv_tgt_data = tsv_tgt_data[:i+line_batch_size]
        #saved_src_tgt_alignments = src_tgt_alignments[:i+line_batch_size]
        with open(out_file, "w") as f_out:
            tsv_output = csv.writer(f_out, delimiter='\t')
            for saved_line in saved_tsv_tgt_data:
                tsv_output.writerow(saved_line)
        #copy the file to the backup file
        copyfile(out_file, backup_file)
#save the final data to file
with open(out_file, "w") as f_out:
    tsv_output = csv.writer(f_out, delimiter='\t')
    for saved_line in tsv_tgt_data:
        tsv_output.writerow(saved_line)
tsv_src.close()


