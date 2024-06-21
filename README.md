## Init
pip install deepl tqdm TTS faster-whisper imageio
## video to srt
1. put video in vt/ori
2. run v2srt.py  
## srt translation
1. modify auth_key = 'deepL_auth_key', put chinese srt file in vt/srt
2. run srt_trans.py  
## srt to speech
1. modify speakers example in vt/ori/speakers  
2. download weight from https://huggingface.co/coqui/XTTS-v2, modify init_xtts()  
set speaker and translated srt file: translate_all_srt(tts_model, speaker, srt_file)
3. run srt2speech.py  
## split background audio
1. pip install spleeter
2. run split_bg.py