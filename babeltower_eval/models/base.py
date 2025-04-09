import os
import re
import gc
from typing import List, Optional, Tuple, Union, Callable
from enum import Enum

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential
from babeltower_eval.utils.schemas import TransDirection
from babeltower_eval.utils.regex import remove_code_block_lines
from babeltower_eval.prompts import CPP2CUDA_TRANSLATION_PROMPT, CUDA2CPP_TRANSLATION_PROMPT, CUDA_CPP_TRANSLATE_TRAIN_SYSTEM


class PromptType(str, Enum):
    TRANS_TRAINED = 'trans_trained'  # with none example
    TRANS = 'trans'                  # with one-shot


class InferModel:
    def __init__(self, model_name: str, mode):
        self.model_name = model_name
        self.mode = mode
        self.trained = True if "trained" in mode else False
        # `None` when self.trained is True
        self.parse_output: Union[None, Callable] = None

    def collect_one(self, system: Optional[str], input: str, sample_num: int = 1) -> str:
        pass

    def collect_batch(self, systems: List[Optional[str]], inputs: List[str], sample_num: int = 1) -> List[str]:
        pass

    def generate_prompt(self, input: str, direction: Optional[TransDirection]) -> Tuple[str, str]:
        """
        TRANS:
            - direction.source == 'CPP' means trans 'CPP' to 'CUDA'
            - direction.source == 'CUDA' means trans 'CUDA' to 'CPP'
        """

        if self.mode == PromptType.TRANS_TRAINED:
            system = CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=direction)

            def _parse_output(output: str) -> List[str]:
                return output
            self.parse_output = _parse_output
            return system, input

        elif self.mode == PromptType.TRANS:
            if direction.source == 'CPP':
                prompt = CPP2CUDA_TRANSLATION_PROMPT.format(cpp_code=input)
            else:
                prompt = CUDA2CPP_TRANSLATION_PROMPT.format(cuda_code=input)

            def _parse_output(output: str) -> str:
                # 首先尝试匹配 [CODE] 和 [/CODE] 之间的内容
                pattern1 = r'\[CODE\](.*?)\[\/CODE\]'
                match1 = re.search(pattern1, output, re.DOTALL)

                if match1:
                    return remove_code_block_lines(match1.group(1).strip())

                # 如果第一个模式匹配失败，尝试匹配 ```cuda 和 ``` 之间的内容
                pattern2 = r'```cuda(.*?)```'
                match2 = re.search(pattern2, output, re.DOTALL)

                if match2:
                    return remove_code_block_lines(match2.group(1).strip())

                # 如果两种模式都匹配失败，抛出异常, 不是抛出异常，证明输出失败了
                return " "
                # raise ValueError("No code block found in the output")

            self.parse_output = _parse_output
            # no system
            return None, prompt
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class OpenAIModel(InferModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI().chat.completions
        self.pipe = self.model.create

    def post_process(self, output: ChatCompletion, n_sample=1) -> Union[str, List[str], List[List[str]]]:
        if (len(output.choices) == 1):
            # collect one
            return [self.parse_output(output.choices[0].message.content)]
        else:
            return [self.parse_output(out.message.content) for out in output.choices]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_one(self, system: Optional[str], input: str, n_samples: int = 1) -> str:
        if system:
            message = [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
        else:
            message = [
                {"role": "user", "content": input},
            ]
        params = {
            "temperature": 0.7,
            "top_p": 1.0,
        }
        if "o1" in self.model_name:
            params.pop("temperature")
        res = self.pipe(
            model=self.model_name,
            n=n_samples,
            messages=message,
            seed=12345,
            **params
        )  # .choices[0].message.content
        ret = self.post_process(res, n_samples)
        return ret
        # return self.parse_output(res)

    def collect_batch(self, systems: List[Optional[str]], inputs: List[str], n_samples: int = 1) -> List[str]:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(
            lambda x: self.collect_one(x[0], x[1], n_samples),
            list(zip(systems, inputs)),
            desc="Collecting"
        )
        ret = []
        for res in results:
            if isinstance(res, str):
                ret.append(res)
            else:
                ret.extend(res)
        return ret


class QWenModel(OpenAIModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).chat.completions
        self.pipe = self.model.create


class DeepSeekModel(OpenAIModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI(
            api_key=os.getenv("SILICON_FLOW"),
            base_url="https://api.siliconflow.cn/v1",
        ).chat.completions
        self.pipe = self.model.create


class InternLmModel(OpenAIModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI(
            api_key=os.getenv("INTERN_LM_API_KEY"),
            base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
        ).chat.completions
        self.pipe = self.model.create


class LocalModel(InferModel):

    def __init__(self, model_name: str, mode: str, tensor_parallel_size: int = 4):
        super().__init__(model_name=model_name, mode=mode)
        from vllm import LLM
        self.model = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.pipe = self.model.chat

    def post_process(self, output, n_sample=1) -> Union[str, List[str], List[List[str]]]:
        if len(output) == 1:
            # collect one
            return self.parse_output(output[0].outputs[0].text)
        else:
            # collect batch
            if n_sample == 1:
                return [self.parse_output(out.outputs[0].text) for out in output]
            else:
                # logger.info(f"{output}")
                return [self.parse_output(out.outputs[i].text) for out in output for i in range(n_sample)]

    def collect_one(self, system: str, input: str, n_sample: int = 1) -> Union[str, List[str]]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=n_sample,
            max_tokens=2048,
            seed=42,
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
        res = self.pipe(
            messages=message,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        return self.post_process(res, n_sample)

    def collect_batch(self, systems: List[str], inputs: List[str], n_sample: int = 1) -> List[str]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=n_sample,
            max_tokens=2048,
            seed=42,
        )
        messages = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
            for system, input in zip(systems, inputs)
        ]
        res = self.pipe(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        return self.post_process(res, n_sample)

    def __del__(self):
        try:
            import torch
            import contextlib

            if torch.cuda.is_available():
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel, destroy_distributed_environment
                )
                destroy_model_parallel()
                destroy_distributed_environment()
                del self.model.llm_engine.model_executor
                del self.model
                with contextlib.suppress(AssertionError):
                    torch.distributed.destroy_process_group()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            del self.model


class ModelFactory:
    OPENAI_MODELS = ["openai", "gpt", "davinci", "o1"]
    QWEN_MODELS = ["qwen"]
    DEEPSEEK_MODELS = ["deepseek"]
    INTERN_LM_MODELS = ["internlm"]

    @staticmethod
    def get_model(model_name: str, mode: PromptType, tensor_parallel_size: int = 4):
        if os.path.exists(model_name):
            return LocalModel(model_name, mode, tensor_parallel_size)
        if any(model in model_name.lower() for model in ModelFactory.OPENAI_MODELS):
            return OpenAIModel(model_name, mode)
        elif any(model in model_name.lower() for model in ModelFactory.QWEN_MODELS):
            return QWenModel(model_name, mode)
        elif any(model in model_name.lower() for model in ModelFactory.DEEPSEEK_MODELS):
            return DeepSeekModel(model_name, mode)
        elif any(model in model_name.lower() for model in ModelFactory.INTERN_LM_MODELS):
            return InternLmModel(model_name, mode)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
