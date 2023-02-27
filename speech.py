import os
import subprocess
import time

import torchaudio
import wave
from TTS.api import TTS
# from espnet2.bin.tts_inference import Text2Speech
# from transformers import AutoModel, AutoTokenizer

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

from util import cleanhtml, log
from removesilence import silenceRemoved

model_name = {
    'english': 'tts_models/en/ljspeech/vits--neon',
    'arabic': '',
    'czech': 'tts_models/cs/cv/vits',
    'german': 'tts_models/de/thorsten/vits',
    'spanish': 'tts_models/es/css10/vits',
    'estonian': 'tts_models/et/cv/vits',
    'finnish': 'tts_models/fi/css10/vits',
    'french': 'tts_models/fr/css10/vits',
    'italian': 'tts_models/it/mai_female/glow-tts',
    'japanese': 'tts_models/ja/kokoro/tacotron2-DDC',
    'korean': '',
    'lithuanian': 'tts_models/lt/cv/vits',
    'latvian': 'tts_models/lv/cv/vits',
    'dutch': 'tts_models/nl/css10/vits',
    'romanian': 'tts_models/ro/cv/vits',
    'turkish': 'tts_models/tr/common-voice/glow-tts',
    'chinese': '',

#    'hindi': 'hi_IN',
    # 'gujarati': 'gu_IN',
    # 'kazakh': 'kk_KZ',
    # 'burmese': 'my_MM',
    # 'nepali': 'ne_NP',
    # 'russian': 'ru_RU',
    # 'sinhala': 'si_LK',
    # 'vietnamese': 'vi_VN',
}

threshold_fastplay = 1.5

def loadData(filename):
    subtitles = []
    with open(filename, encoding='utf-8-sig') as f:
        lines = f.readlines()
        idx = 1
        for el in lines:
            el = el.strip()
            try:
                value = int(el)
            except ValueError:
                value = None

            if value is not None and idx == value:
                subt = {
                    'idx': idx,
                    'content': []
                }
                subtitles.append(subt)
                idx = idx+1
            else:
                curr = subtitles[idx-2]
                if "-->" in el:
                    arr = el.split("-->")
                    curr['startTime'] = arr[0].strip()
                    curr['endTime'] = arr[1].strip()
                elif len(el) == 0:
                    continue
                else:
                    strg = cleanhtml(el)
                    # print('strg:', strg)
                    if strg[0] != '\"' and strg[-1] == '\"':
                        strg = strg.strip('\"')
                    if strg[0] == '\"' and strg[-1] != '\"':
                        strg = strg.strip('\"')
                    curr['content'].append(strg)
    return subtitles

def generateAudio(subtitles, content, language):
    global model_name
    subprocess.call(['mkdir', '-p', f'media/{content}/audio/{language}'])

    if language == 'korean':
        model_name = "imdanboy/kss_tts_train_jets_raw_phn_korean_cleaner_korean_jaso_train.total_count.ave"
        text2speech = Text2Speech.from_pretrained(model_name)

        for s in subtitles:
            path = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
            article = ' '.join(s['content'])
            # print(article)
            if len(article) < 5:
                article = "!"+article
            try:
                speech = text2speech(article)["wav"]
                audio = speech.view(1,-1) 
                torchaudio.save(path, audio, 22050)
            except RuntimeError as err:
                print('Error:', err)

    elif language == 'arabic':
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/tts_transformer-ar-cv7",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        generator = task.build_generator([model], cfg)

        for s in subtitles:
            path = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
            article = ' '.join(s['content'])
            sample = TTSHubInterface.get_model_input(task, article)
            wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
            wav = wav.view(1,-1)
            torchaudio.save(path, wav, 22050)
            if len(article) > 6:
                tmpFile = f"media/{content}/audio/{language}/tmp.wav"
                subprocess.call(['ffmpeg', '-y', '-i', path, '-ar', '16000', tmpFile])
                silenceRemoved(tmpFile, path)
        subprocess.call(['rm', tmpFile])

    elif language == 'chinese':
        text2speech = Text2Speech.from_pretrained("espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best")
        for s in subtitles:
            path = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
            article = ' '.join(s['content'])
            speech = text2speech(article)["wav"]
            audio = speech.view(1,-1) 
            torchaudio.save(path, audio, 22050)

    else:
        model = model_name[language]
        tts = TTS(model)
        for s in subtitles:
            path = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
            article = ' '.join(s['content'])
            # print(article)
            tts.tts_to_file(text=article, file_path=path)
            if len(article) > 6:
                tmpFile = f"media/{content}/audio/{language}/tmp.wav"
                subprocess.call(['ffmpeg', '-y', '-i', path, '-ar', '16000', tmpFile])
                silenceRemoved(tmpFile, path)
        subprocess.call(['rm', tmpFile])

def convertToSeconds(s):
    h = int(s.split(':')[0])
    m = int(s.split(':')[1])
    s = s.split(':')[2]
    s = float(s.replace(',', '.'))
    seconds = h*3600 + m*60+ s
    return seconds

def changeAudioLength(subtitles, content, language):
    subprocess.call(['mkdir', '-p', f'media/{content}/audio_modified/{language}'])
    for s in subtitles:
        # print(s['idx'])
        path = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
        if not os.path.exists(path):
            continue
        try :
            with wave.open(path,'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
        except wave.Error:
            metadata = torchaudio.info(path)
            frames = metadata.num_frames
            rate = metadata.sample_rate
            duration = frames / float(rate)

        start = convertToSeconds(s['startTime'])
        end = convertToSeconds(s['endTime'])
        durationTarget = end - start
        tempo = min(threshold_fastplay, max(1.0, duration/durationTarget))
        tmpFile = f"media/{content}/audio_modified/{language}/tmp_{content}.wav"
        outputFilename = f"media/{content}/audio_modified/{language}/{content}_{language}_script_{s['idx']:03}.wav"
        subprocess.call(['ffmpeg', '-y', '-i', path, '-codec:a', 'libmp3lame', '-filter:a', f'atempo={tempo}', '-b:a', '320K', tmpFile])
        subprocess.call(['ffmpeg', '-y', '-i', tmpFile, '-ar', '44100', outputFilename])
    subprocess.call(['rm', tmpFile])

def mergeAudio(subtitles, content, language):
    # with wave.open('avatar.wav', 'r') as f:
    #     frames = f.getnframes()
    #     rate = f.getframerate()
    #     totalLength = frames / float(rate)
    lst = subtitles[-1]
    totalLength = convertToSeconds(lst['endTime'])+5

    # print(totalLength)
    pathTmpAudioIn = f'media/{content}/{content}_{language}_in.wav'
    pathTmpAudioOut = f'media/{content}/{content}_{language}_out.wav'
    subprocess.call(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100', '-t', '%.03f'%totalLength, '-q:a', '9', '-acodec', 'libmp3lame', pathTmpAudioIn])

    delay = -1
    for idx, s in enumerate(subtitles):
        # print(s['idx'])
        path = f"media/{content}/audio_modified/{language}/{content}_{language}_script_{s['idx']:03}.wav"
        with wave.open(path,'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        startTime = convertToSeconds(s['startTime'])
        endTime = convertToSeconds(s['endTime'])
        if delay > 0:
            prev = subtitles[idx-1]
            endTimePrev = convertToSeconds(prev['endTime'])
            if endTimePrev+delay > startTime:
                # print(f'#####delay: {endTimePrev+delay-startTime}')
                startTime = endTimePrev+delay
                originalFile = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
                try :
                    with wave.open(originalFile,'r') as f:
                        fr_orig = f.getnframes()
                        r_orig = f.getframerate()
                        du_orig = fr_orig / float(r_orig)
                except wave.Error:
                    metadata = torchaudio.info(originalFile)
                    fr_orig = metadata.num_frames
                    r_orig = metadata.sample_rate
                    du_orig = frames / float(r_orig)
                du_tgt = endTime-startTime
                tempo = min(threshold_fastplay, max(1.0, du_orig/du_tgt))
                tmpFile = f"media/{content}/audio_modified/{language}/tmp_fast.wav"
                outputFile = f"media/{content}/audio_modified/{language}/tmp_fast_modified.wav"
                subprocess.call(['ffmpeg', '-y', '-i', originalFile, '-codec:a', 'libmp3lame', '-filter:a', f'atempo={tempo}', '-b:a', '320K', tmpFile])
                subprocess.call(['ffmpeg', '-y', '-i', tmpFile, '-ar', '44100', outputFile])
                with wave.open(outputFile,'r') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                path = outputFile

        length = endTime-startTime
        if duration+0.2 < length:
            startTime = startTime+(length-duration)/2
            endTime = endTime-(length-duration)/2
        if duration > length:
            delay = duration-length
        else:
            delay = -1
        startTime = "%.03f"%startTime
        endTime = "%.03f"%endTime
        filterString = f"[0]atrim=0:{startTime}[Apre];[0]atrim={endTime},asetpts=PTS-STARTPTS[Apost];[Apre][1][Apost]concat=n=3:v=0:a=1"
        # print(filterString)
        subprocess.call(['ffmpeg', '-y', '-i', pathTmpAudioIn, '-i', path,
            '-filter_complex', filterString, pathTmpAudioOut])
        subprocess.call(['mv', pathTmpAudioOut, pathTmpAudioIn])

    subprocess.call(['mv', pathTmpAudioIn, f'media/{content}/{content}_{language}.wav'])
    subprocess.call(['ffmpeg', '-y', '-i', f'media/{content}/{content}_{language}.wav', f'media/{content}/{content}_{language}.m4a'])
    subprocess.call(['rm', tmpFile])
    subprocess.call(['rm', outputFile])

def mergeAudio(subtitles, content, language):
    # with wave.open('avatar.wav', 'r') as f:
    #     frames = f.getnframes()
    #     rate = f.getframerate()
    #     totalLength = frames / float(rate)
    lst = subtitles[-1]
    totalLength = convertToSeconds(lst['endTime'])+5

    # print(totalLength)
    pathTmpAudioIn = f'media/{content}/{content}_{language}_in.wav'
    pathTmpAudioOut = f'media/{content}/{content}_{language}_out.wav'
    subprocess.call(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100', '-t', '%.03f'%totalLength, '-q:a', '9', '-acodec', 'libmp3lame', pathTmpAudioIn])

    delay = -1
    for idx, s in enumerate(subtitles):
        # print(s['idx'])
        path = f"media/{content}/audio_modified/{language}/{content}_{language}_script_{s['idx']:03}.wav"
        with wave.open(path,'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        startTime = convertToSeconds(s['startTime'])
        endTime = convertToSeconds(s['endTime'])
        if delay > 0:
            prev = subtitles[idx-1]
            endTimePrev = convertToSeconds(prev['endTime'])
            if endTimePrev+delay > startTime:
                # print(f'#####delay: {endTimePrev+delay-startTime}')
                startTime = endTimePrev+delay
                originalFile = f"media/{content}/audio/{language}/{content}_{language}_script_{s['idx']:03}.wav"
                try :
                    with wave.open(originalFile,'r') as f:
                        fr_orig = f.getnframes()
                        r_orig = f.getframerate()
                        du_orig = fr_orig / float(r_orig)
                except wave.Error:
                    metadata = torchaudio.info(originalFile)
                    fr_orig = metadata.num_frames
                    r_orig = metadata.sample_rate
                    du_orig = frames / float(r_orig)
                du_tgt = endTime-startTime
                tempo = min(threshold_fastplay, max(1.0, du_orig/du_tgt))
                tmpFile = f"media/{content}/audio_modified/{language}/tmp_fast.wav"
                outputFile = f"media/{content}/audio_modified/{language}/tmp_fast_modified.wav"
                subprocess.call(['ffmpeg', '-y', '-i', originalFile, '-codec:a', 'libmp3lame', '-filter:a', f'atempo={tempo}', '-b:a', '320K', tmpFile])
                subprocess.call(['ffmpeg', '-y', '-i', tmpFile, '-ar', '44100', outputFile])
                with wave.open(outputFile,'r') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                path = outputFile

        length = endTime-startTime
        if duration+0.2 < length:
            startTime = startTime+(length-duration)/2
            endTime = endTime-(length-duration)/2
        if duration > length:
            delay = duration-length
        else:
            delay = -1
        startTime = "%.03f"%startTime
        endTime = "%.03f"%endTime
        filterString = f"[0]atrim=0:{startTime}[Apre];[0]atrim={endTime},asetpts=PTS-STARTPTS[Apost];[Apre][1][Apost]concat=n=3:v=0:a=1"
        # print(filterString)
        subprocess.call(['ffmpeg', '-y', '-i', pathTmpAudioIn, '-i', path,
            '-filter_complex', filterString, pathTmpAudioOut])
        subprocess.call(['mv', pathTmpAudioOut, pathTmpAudioIn])

    subprocess.call(['mv', pathTmpAudioIn, f'media/{content}/{content}_{language}.wav'])
    subprocess.call(['ffmpeg', '-y', '-i', f'media/{content}/{content}_{language}.wav', f'media/{content}/{content}_{language}.m4a'])
    subprocess.call(['rm', tmpFile])
    subprocess.call(['rm', outputFile])

def makeSpeech(content, language):
    arr = loadData(f'media/{content}/subtitle/{content}_{language}.txt')
    generateAudio(arr, content, language)
    changeAudioLength(arr, content, language)
    mergeAudio(arr, content, language)

def isSupported(language):
    if language in model_name.keys():
        return True
    else:
        return False

