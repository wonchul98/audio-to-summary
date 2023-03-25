from speechbrain.pretrained import SepformerSeparation as separator
import os
import torchaudio
from pydub import AudioSegment

# Load the speaker separation model
model = separator.from_hparams(source="speechbrain/sepformer-libri2mix", savedir='pretrained_models/sepformer-libri2mix',run_opts={"device":"cuda"})

# Define the folder paths
folder_path = './drive/MyDrive/kspon'
new_folder_path = './drive/MyDrive/sep_kspon'

# Create the new folder for the separated audio files
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# Separate each audio file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pcm'):
        # Load the PCM file and convert it to a temporary WAV file
        pcm_path = os.path.join(folder_path, filename)
        tmp_wav_path = os.path.join(folder_path, f"{filename.split('.')[0]}.wav")
        AudioSegment.from_file(pcm_path, format="raw", sample_width=2, frame_rate=16000, channels=1).export(tmp_wav_path, format="wav")

        # Separate the temporary WAV file
        est_sources = model.separate_file(path=tmp_wav_path)

        # Save the separated sources to new WAV files
        for k in range(2):
            out_filename = f"{filename.split('.')[0]}_{k}.wav"
            out_path = os.path.join(new_folder_path, out_filename)
            torchaudio.save(out_path, est_sources[:, :, k].detach().cpu(), 8000)

        # Remove the temporary WAV file
        os.remove(tmp_wav_path)

    elif filename.endswith('.wav'):
        # Separate the WAV file
        path = os.path.join(folder_path, filename)
        est_sources = model.separate_file(path=path)

        # Save the separated sources to new WAV files
        for k in range(2):
            out_filename = f"{filename.split('.')[0]}_{k}.wav"
            out_path = os.path.join(new_folder_path, out_filename)
            torchaudio.save(out_path, est_sources[:, :, k].detach().cpu(), 8000)