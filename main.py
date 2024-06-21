# Import necessary modules
from llama import Llama as LL
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as generative_ai
from openai import OpenAI
from typing import Union, List


def fetch_api_key(file_path, multiple_keys=False):
    if multiple_keys:
        api_keys = []
        with open(file_path, "r", encoding="utf-8") as file:
            api_keys = file.readlines()
            api_keys = [key.strip() for key in api_keys]
        return api_keys

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


class LLModel:
    def __init__(self, model_version, ckpt_path, max_length=2048, temp=0.1):
        """
        Model Versions:
        Meta-Llama-3-70B-Instruct
        Meta-Llama-3-70B
        Meta-Llama-3-8B-Instruct
        Meta-Llama-3-8B
        """
        self.model_version = model_version
        self.ckpt_path = ckpt_path  # Unused variable
        self.tokenizer = AutoTokenizer.from_pretrained(model_version)
        self.max_length = max_length
        self.temp = temp

        self.model = AutoModelForCausalLM.from_pretrained(
            model_version,
            load_in_4bit=True,
            device_map="auto",
        )

    def query(self, user_input: str, categories: Union[None, List[str]] = None) -> str:
        input_messages = [
            {"role": "user", "content": user_input},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            input_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        end_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(""),
        ]
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_length,
            eos_token_id=end_tokens,
            temperature=self.temp,
        )

        if categories is None:
            result = output_ids[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(result, skip_special_tokens=True)

        result = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(result, skip_special_tokens=True)


class GPTChat:
    def __init__(self, model_version, api_key_file, max_length=4096, temp=1e-6):
        """
        Model Versions:
        gpt-4-0125-preview, gpt-4-1106-preview
        gpt-4, gpt-4-32k
        gpt-3.5-turbo-0125, gpt-3.5-turbo-instruct
        """
        self.model_version = model_version
        self.api_key = fetch_api_key(api_key_file)
        self.max_length = max_length
        self.temp = temp

    def query(self, user_input: str, categories: Union[None, List[str]] = None) -> str:
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model=self.model_version,
            max_tokens=self.max_length,
            temperature=self.temp,
            logprobs=True,
            top_logprobs=20,
        )

        if categories is None:
            return response.choices[0].message.content

        class_logprobs = {}
        for candidate in categories:
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            max_logprob = -float("inf")
            for logprob_obj in top_logprobs:
                if candidate.lower().startswith(logprob_obj.token.strip().lower()):
                    if logprob_obj.logprob > max_logprob:
                        max_logprob = logprob_obj.logprob
            class_logprobs[candidate] = max_logprob

        return max(class_logprobs, key=class_logprobs.get)


class GeminiAI:
    def __init__(self, model_version, api_key_file):
        """
        Model Versions:
        gemini-1.0-pro, gemini-1.0-pro-001
        gemini-1.0-pro-latest, gemini-1.0-pro-vision-latest
        gemini-pro, gemini-pro-vision
        """
        self.model_version = model_version
        self.api_keys = fetch_api_key(api_key_file, multiple_keys=True)
        if not self.api_keys:
            raise ValueError("No API keys found.")
        self.current_key_index = 0

    def query(self, user_input):
        safety_categories = [
            "HARM_CATEGORY_DANGEROUS",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
        safety_policies = [
            {"category": category, "threshold": "BLOCK_NONE"}
            for category in safety_categories
        ]

        generative_ai.configure(api_key=self.api_keys[self.current_key_index])
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        model = generative_ai.GenerativeModel(self.model_version)
        response = model.generate_content(user_input, safety_settings=safety_policies)
        
        if not response.candidates:
            return "None returned from Gemini"
        return response.text


class LanguageModel:
    def __init__(self, model_name: str, model_version: str, path: str):
        """'path' is the API key path or the model file path."""
        self.model_name = model_name.lower()
        self.model = None

        if self.model_name == "chatgpt":
            self.model = GPTChat(model_version, path)
        elif self.model_name == "gemini":
            self.model = GeminiAI(model_version, path)
        elif self.model_name == "llama":
            self.model = LLModel(model_version, path)
        elif self.model_name == "test":
            self.model = None
        else:
            raise ValueError(f"Unsupported language model: {self.model_name}")

    def generate_response(self, user_input: str, categories: Union[None, List[str]] = None) -> str:
        if self.model_name in ["chatgpt", "llama"]:
            return self.model.query(user_input, categories)
        elif self.model_name == "gemini":
            return self.model.query(user_input)
        elif self.model_name == "test":
            return "<answer>Test response</answer>"
        else:
            raise ValueError(f"Unsupported language model: {self.model_name}")
