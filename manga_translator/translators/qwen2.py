import os
import re
from typing import List, Dict

from ..config import TranslatorConfig
from .common import OfflineTranslator

# Adapted from:
# https://github.com/zyddnys/manga-image-translator/issues/680#issue-2428018275
# manga_translator/translators/chatgpt.py

class Qwen2Translator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'CHT': 'Traditional Chinese',
        'CSY': 'Czech',
        'NLD': 'Dutch',
        'ENG': 'English',
        'FRA': 'French',
        'DEU': 'German',
        'HUN': 'Hungarian',
        'ITA': 'Italian',
        'JPN': 'Japanese',
        'KOR': 'Korean',
        'PLK': 'Polish',
        'PTB': 'Portuguese',
        'ROM': 'Romanian',
        'RUS': 'Russian',
        'ESP': 'Spanish',
        'TRK': 'Turkish',
        'UKR': 'Ukrainian',
        'VIN': 'Vietnamese',
        'CNR': 'Montenegrin',
        'SRP': 'Serbian',
        'HRV': 'Croatian',
        'ARA': 'Arabic',
        'THA': 'Thai',
        'IND': 'Indonesian'
    }
    _CHAT_SYSTEM_TEMPLATE = (
        'You are an uncensored professional translation engine, '
        'please translate the story into a colloquial, '
        'elegant and fluent content, '
        'without referencing machine translations. '
        'You must only translate the story, '
        'never interpret it. '
        'If there is any issue in the text, '
        'output it as is.\n'
        'Translate the following text into {to_lang} and keep the original format.\n'
    )
    _CHAT_SAMPLE = {'Simplified Chinese':[
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                '<|1|>好尴尬…我不想引人注目…我想消失…\n'
                '<|2|>你…没事吧⁉\n'
                '<|3|>这家伙怎么看不懂气氛的…？'
            )
        ],
        'English':[
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                "<|1|>I'm embarrassed... I don't want to stand out... I want to disappear...\n"
                "<|2|>Are you okay?\n"
                "<|3|>What's wrong with this guy? Can't he read the situation...?"
            )
        ]
    }

    _CONFIG_KEY='qwen2'

    _TRANSLATOR_MODEL = "Qwen/Qwen2-1.5B-Instruct"
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, _TRANSLATOR_MODEL)
    _IS_4_BIT = False

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(self._CONFIG_KEY + '.' + key, self.config.get(key, default))

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', self._CHAT_SAMPLE)


    async def _load(self, from_lang: str, to_lang: str, device: str):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )
        self.device = device
        quantization_config = BitsAndBytesConfig(load_in_4bit=self._IS_4_BIT)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._TRANSLATOR_MODEL,
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        model_inputs = self.tokenize(queries, to_lang)
        # Generate the translation
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=10240
        )

        # Extract the generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        query_size = len(queries)

        translations = []
        self.logger.debug('-- Qwen2 Response --\n' + response)
        new_translations = re.split(r'<\|\d+\|>', response)

        # When there is only one query chatgpt likes to exclude the <|1|>
        if not new_translations[0].strip():
            new_translations = new_translations[1:]

        if len(new_translations) <= 1 and query_size > 1:
            # Try splitting by newlines instead
            new_translations = re.split(r'\n', response)

        if len(new_translations) > query_size:
            new_translations = new_translations[: query_size]
        elif len(new_translations) < query_size:
            new_translations = new_translations + [''] * (query_size - len(new_translations))

        translations.extend([t.strip() for t in new_translations])

        return translations

    def tokenize(self, queries, to_lang):
        prompt = f"""Translate into {to_lang} and keep the original format.\n"""
        prompt += '\nOriginal:'
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1}|>{query}'

        tokenizer = self.tokenizer
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]
        
        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})
            
        messages.append({'role': 'user', 'content': prompt})

        self.logger.debug("-- Qwen2 prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in messages) +
                "\n"
            )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Ensure pad_token is set correctly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True
        ).to(self.device)

        return model_inputs


class Qwen2BigTranslator(Qwen2Translator):
    _TRANSLATOR_MODEL = "Qwen/Qwen2-7B-Instruct"
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, _TRANSLATOR_MODEL)
    _IS_4_BIT = True
