from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
torchaudio.set_audio_backend('pydub')
model = separator.from_hparams(source="speechbrain/sepformer-libri3mix", savedir='pretrained_models/sepformer-libri3mix',run_opts={"device":"cuda"})
est_sources = model.separate_file(path='../data/hamburger.wav') 

torchaudio.save("ham1.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("ham2.wav", est_sources[:, :, 1].detach().cpu(), 8000)
torchaudio.save("ham3.wav", est_sources[:, :, 2].detach().cpu(), 8000)