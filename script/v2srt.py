import whisper
import os
import datetime, time
from zhconv import convert
from tqdm import tqdm
import imageio
from whisper.utils import get_writer
from faster_whisper import WhisperModel
from datetime import timedelta

def find_files(path, suffix):
    mp4_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.' + suffix):
                mp4_files.append(os.path.abspath(os.path.join(root, file)))
    return mp4_files

def format_time(seconds):
    return str(timedelta(seconds=seconds)).split('.')[0] + ",000"

def v2srt_whisper(file_path, output_dir):
    mp4_videos = find_files(file_path, suffix='mp4')

    model = whisper.load_model('large-v3')
    initial_prompt = "以下是普通话的句子。"
    for video in tqdm(mp4_videos):
        # xxx.mp4 --> xxx. + srt
        video_name = video.split('/')[-1].split('.')[0]
        print('v2srt Processing: {}'.format(video_name))
        res = model.transcribe(video, fp16=False, language='Chinese')

        writer = get_writer('srt', output_dir)
        writer(res, video_name)

def v2srt_faster_whisper(file_path, output_dir):
    mp4_videos = find_files(file_path, suffix='mp4')
    model = WhisperModel('large-v3',
                         compute_type="float32",
                         num_workers=1)
    for video in tqdm(mp4_videos):
        video_name = video.split('/')[-1].split('.')[0]
        print('v2srt Processing: {}'.format(video_name))
        segments, info = model.transcribe(video,
                                      beam_size=5,
                                      best_of=1,
                                      temperature=0,
                                      vad_filter=True,
                                      vad_parameters=dict(
                                          min_silence_duration_ms=1000,
                                          max_speech_duration_s=2
                                      ),
                                      word_timestamps=True,
                                      language='zh',
                                      initial_prompt=None)
        srt_output = ''
        for index, segment in enumerate(segments, start=1):
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            text = convert(segment.text, 'zh-cn')
            srt_output += f"{index}\n{start_time} --> {end_time}\n{text}\n\n"
        with open(os.path.join(output_dir,video_name+'.srt'), "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_output)

if __name__ == '__main__':
    file_path = r'/home/dh/video_trans/ori_video'
    output_dir = r'/home/dh/video_trans/srt'
    # v2srt_whisper(file_path, output_dir)
    v2srt_faster_whisper(file_path, output_dir)