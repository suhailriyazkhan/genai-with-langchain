from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceChatClient:
    def __init__(self, repo_id="google/gemma-2-2b-it", temperature=1.0, max_tokens=256):
        self.repo_id = repo_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._create_model()

    def _create_model(self):
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            task="text-generation",
            temperature=self.temperature,
            max_new_tokens=self.max_tokens
        )
        self.model = ChatHuggingFace(llm=self.llm)

    def set_model(self, repo_id):
        self.repo_id = repo_id
        self._create_model()

    def set_temperature(self, temperature):
        self.temperature = temperature
        self._create_model()

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        self._create_model()

# Usage
if __name__ == "__main__":
    client = HuggingFaceChatClient()
    # client.set_model("google/gemma-7b-it")
    client.set_temperature(0.7)
    client.set_max_tokens(128)
    answer = client.model.invoke("What is the capital of India?")
    print(answer)