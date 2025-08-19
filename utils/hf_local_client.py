from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/Python Projects/gen-ai-projects/gen-ai-with-langchain/huggingface_cache'


class HuggingFaceLocalChatClient:
    def __init__(self, model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', temperature=0.5, max_new_tokens=100):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._create_model()

    def _create_model(self):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.model_id,
            task='text-generation',
            pipeline_kwargs=dict(
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens
            )
        )
        self.model = ChatHuggingFace(llm=self.llm)