import deepl
import re
import os
from tqdm import tqdm
auth_key = 'deepL_auth_key'

def translate_srt(input_file, output_file, auth_key, source_lang, target_lang):
    # Initialize the DeepL translator
    translator = deepl.Translator(auth_key)
    
    with open(input_file, 'r', encoding='utf-8') as file:
        srt_content = file.read()
 
    # Regular expression to match the SRT subtitle blocks
    srt_blocks = re.split(r'\n(?=\d+\n)', srt_content.strip())
    
    translated_srt = ""
    for block in srt_blocks:
        # Split the block into its components: index, timestamp, and text
        lines = block.strip().split('\n')
        index = lines[0]
        timestamp = lines[1]
        text = '\n'.join(lines[2:])
        
        # Translate the text
        if text:
            print(text)
            result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
            translated_text = result.text
        else:
            translated_text = ""
        
        translated_srt += f"{index}\n{timestamp}\n{translated_text}\n\n"
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translated_srt)

def translate_all_srt(input_dir, output_dir, auth_key, source_lang='ZH', target_lang='EN-US'):


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith('.srt'):
            translated_file = file.replace('.srt', f'_{target_lang}.srt')
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, translated_file)
            translate_srt(input_file, output_file, auth_key, source_lang, target_lang)

if __name__ == '__main__':
    input_file = '/home/dh/video_trans/srt'
    output_file = '/home/dh/video_trans/srt_translated'
    translate_all_srt(input_file, output_file, auth_key="47683fff-2408-429b-bb0c-d0d1f386ca83:fx")
