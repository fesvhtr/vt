import subprocess
import os



audio_path = '/home/dh/video_trans/ori/1109_0_1800.WAV'

output_path = '/home/dh/video_trans/out'
audio_name = audio_path.split('/')[-1].split('.')[0]
output_name = f'{audio_name}_bg'



subprocess.run(['spleeter', 'separate', '-o', output_path, audio_path])

accompaniment_path = os.path.join(output_path, f'{audio_name}_accompaniment.wav')

print(f"Accompaniment saved to: {accompaniment_path}")