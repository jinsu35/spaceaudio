import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

from util import cleanhtml

languageCode = {
    'arabic': 'ar_AR',
    'czech': 'cs_CZ',
    'german': 'de_DE',
    # 'english': 'en_XX',
    'spanish': 'es_XX',
    'estonian': 'et_EE',
    'finnish': 'fi_FI',
    'french': 'fr_XX',
    # 'gujarati': 'gu_IN',
    # 'hindi': 'hi_IN',
    'italian': 'it_IT',
    'japanese': 'ja_XX',
    # 'kazakh': 'kk_KZ',
    'korean': 'ko_KR',
    'lithuanian': 'lt_LT',
    'latvian': 'lv_LV',
    # 'burmese': 'my_MM',
    # 'nepali': 'ne_NP',
    'dutch': 'nl_XX',
    'romanian': 'ro_RO',
    # 'russian': 'ru_RU',
    # 'sinhala': 'si_LK',
    'turkish': 'tr_TR',
    # 'vietnamese': 'vi_VN',
    'chinese': 'zh_CN'
}

def writeTranscript(content, source, language):
    src_lang = "en_XX"
    # source = f'media/{content}/subtitle/{content}_{src_lang}.txt'

    script = []

    with open(source, encoding='utf-8-sig') as f:
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
                script.append(subt)
                idx = idx+1
            else:
                curr = script[idx-2]
                if "-->" in el:
                    arr = el.split("-->")
                    curr['startTime'] = arr[0].strip()
                    curr['endTime'] = arr[1].strip()
                else:
                    curr['content'].append(cleanhtml(el))

    tgt_lang = languageCode[language]
    destination = f'media/{content}/subtitle/{content}_{language}.txt'

    if tgt_lang == 'es_XX':
        transformer_model = "Helsinki-NLP/opus-mt-en-es"
        model = MarianMTModel.from_pretrained(transformer_model).to(torch.device('mps'))
        tokenizer = MarianTokenizer.from_pretrained(transformer_model)
    elif tgt_lang == 'ar_AR':
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ar")
    elif tgt_lang == 'tr_TR':
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")
    elif tgt_lang == 'ro_RO':
        sentences = [' '.join(x['content']) for x in script]
        modelname = "t5-3b" # or t5-base, t5-3b, t5-11b
        tokenizer = T5Tokenizer.from_pretrained(modelname)
        model = T5ForConditionalGeneration.from_pretrained(modelname)

        task_prefix = "translate English to Romanian: "
        # use different length sentences to test batching

        inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
        )

        outputSentences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        # print(outputSentences)
    else:
        transformer_model = "facebook/mbart-large-50-many-to-many-mmt"
        model = MBartForConditionalGeneration.from_pretrained(transformer_model).to(torch.device('mps'))
        tokenizer = MBart50TokenizerFast.from_pretrained(transformer_model)
        tokenizer.src_lang = src_lang

    with open(destination, 'w', encoding='utf-8') as file:
        for s in script:
            file.write(f"{s['idx']}")
            file.write('\n')
            file.write(f"{s['startTime']} --> {s['endTime']}")
            file.write('\n')
            article_i = ' '.join(s['content'])
            # print(article_i)
            if tgt_lang == 'es_XX':
                encoded_i = tokenizer(article_i, return_tensors="pt").to('mps')
                generated_tokens = model.generate(**encoded_i)
                article_o = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            elif tgt_lang == 'ar_AR' or tgt_lang == 'tr_TR':
                # print('hello')
                article_o = [x['translation_text'] for x in pipe(f">>ara<< {article_i}")]
            elif tgt_lang == 'ro_RO':
                idx = s['idx']
                sentence = outputSentences[idx-1]
                article_o = [sentence]
            else:
                encoded_i = tokenizer(article_i, return_tensors="pt").to('mps')
                generated_tokens = model.generate(**encoded_i, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
                article_o = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            file.write(article_o[0])
            file.write('\n')
            file.write('\n')

