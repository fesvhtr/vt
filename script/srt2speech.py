import srt
import torch
from TTS.api import TTS
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import shutil
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
# huggingface-cli download --resume-download coqui/XTTS-v2 --local-dir /home/dh/video_trans/weights

device = "cuda" if torch.cuda.is_available() else "cpu"
speakers = {
    'default':'/home/dh/video_trans/ori/speakers/en_sample.wav',
    'yjk':'/home/dh/video_trans/ori/speakers/yjk.wav',
}

def extract_audio_segment(input_file, output_file, start_time, end_time):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file)
    # Calculate start and end time in milliseconds
    start_time_ms = start_time * 1000
    end_time_ms = end_time * 1000
    # Extract the segment
    extracted_segment = audio[start_time_ms:end_time_ms]
    # Export the extracted segment to a new file
    extracted_segment.export(output_file, format="mp3")

def text2speech(tts, input_text, speaker_name, file_path):
    tts.tts_to_file(text=input_text, speaker_wav=speakers[speaker_name], language="en", file_path=file_path)

def parse_srt(srt_file):
    with open(srt_file, 'r', encoding='utf-8') as file:
        content = file.read()
    subs = list(srt.parse(content))
    return subs

def adjust_audio_duration(audio_segment, target_duration_ms):
    current_duration_ms = len(audio_segment)
    if current_duration_ms == target_duration_ms:
        return audio_segment
    elif current_duration_ms > target_duration_ms:
        speed_change = current_duration_ms / target_duration_ms
        return audio_segment.speedup(playback_speed=speed_change)
    else:
        return audio_segment
    
def text2speech_speedup(tts_model, input_text, speaker_name, speed, file_path):
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[speakers[speaker_name]])
    out = tts_model.inference(
        input_text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
        speed = speed # Add custom parameters here
    )
    torchaudio.save(file_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)



def srt2speech_speedup_v1(tts, speaker_name, srt_file,default_speed=1.1):
    subs = parse_srt(srt_file)
    srt_filename = Path(srt_file).stem
    output_dir = Path("/home/dh/video_trans/out")
    temp_output_dir = Path("/home/dh/video_trans/out") / srt_filename
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    audio_segments = []

    for i, sub in tqdm(enumerate(subs), total=len(subs)):
        # Generate temporary audio file path
        temp_audio_path = temp_output_dir / f"output_{i}.wav"
        # Calculate the target duration in milliseconds
        target_duration_ms = (sub.end - sub.start).total_seconds() * 1000
        # Calculate the start time in milliseconds
        start_time_ms = sub.start.total_seconds() * 1000

        # Generate speech for each subtitle
        text2speech_speedup(tts, sub.content, speaker_name, default_speed, temp_audio_path)
        # Load the generated audio file
        audio_segment = AudioSegment.from_wav(temp_audio_path)
        

        # Check for overlap with the next subtitle
        if i < len(subs) - 1:
            next_start_time_ms = subs[i + 1].start.total_seconds() * 1000
            if start_time_ms + len(audio_segment) > next_start_time_ms:
                print(f'Start time: {start_time_ms}, End time: {start_time_ms + len(audio_segment)}, Next start time: {next_start_time_ms}')
                overlap_duration = (start_time_ms + len(audio_segment)) - next_start_time_ms
                target_duration_ms = len(audio_segment) - overlap_duration
                speed =  len(audio_segment) / target_duration_ms
                print(f'Overlap detected: {overlap_duration} ms, Adjusting speed to {speed}')
                # Ensure the speed is at least 1.5x
                speed = max(1.5, speed)
                text2speech_speedup(tts, sub.content, speaker_name, speed, temp_audio_path)
                audio_segment = AudioSegment.from_wav(temp_audio_path)      
            adjusted_audio_segment = audio_segment
        
        # Append the audio segment with its start time
        audio_segments.append((start_time_ms, adjusted_audio_segment))

    # Determine the total duration of the final audio
    total_duration = max(start_time_ms + len(audio_segment) for start_time_ms, audio_segment in audio_segments)

    # Start with a silent audio segment of the required total duration
    final_audio = AudioSegment.silent(duration=total_duration)

    for start_time_ms, audio_segment in audio_segments:
        final_audio = final_audio.overlay(audio_segment, position=start_time_ms)

    # Export the final combined audio
    output_file = output_dir / f"{srt_filename}.wav"
    final_audio.export(output_file, format="wav")
    shutil.rmtree(temp_output_dir)

def init_xtts():
    print("Loading model...")
    config = XttsConfig()
    config.load_json("/home/dh/video_trans/weights/xtts_v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/dh/video_trans/weights/xtts_v2", use_deepspeed=False)
    model.cuda()
    return model

if __name__ == '__main__':

    
    tts_model = init_xtts()
    srt2speech_speedup_v1(tts_model, 'yjk', '/home/dh/video_trans/srt_translated/1109_0_1800_EN-US.srt')