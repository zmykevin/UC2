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


source_label_file = "img_label_objects.txt"
tgt_lan = "cs"
tgt_label_file = "img_label_objects_{}.txt".format(tgt_lan)
batch_size = 100

source_labels = []
tgt_labels = []
with open(source_label_file, "r") as f:
    for line in f:
        source_labels.append(line.strip())

for i in tqdm(range(0, len(source_labels), batch_size)):
    batch_upper_limit = i+batch_size if i+batch_size < len(source_labels) else len(source_labels)
    current_src_batch = source_labels[i:batch_upper_limit]
    current_batch_size = len(current_src_batch)

    current_translation_batch = msft_translate(current_src_batch, langs=tgt_lan, batch_size=current_batch_size)
    for translated_data in current_translation_batch:
        assert translated_data['translations'][0].get('text', None)
        tgt_labels.append(translated_data['translations'][0].get('text', None))
#write the tgt_labels to outputfile
with open(tgt_label_file, 'w') as f_out:
    for label in tgt_labels:
        f_out.write(label+'\n')