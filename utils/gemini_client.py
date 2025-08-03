from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class GoogleGenAIChatClient:
    def __init__(self, model="gemini-1.5-flash-latest", temperature=1.0, max_tokens=256):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._create_model()

    def _create_model(self):
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def set_model(self, model_name):
        self.model_name = model_name
        self._create_model()

    def set_temperature(self, temperature):
        self.temperature = temperature
        self._create_model()

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        self._create_model()

# Usage
if __name__ == "__main__":
    # Create client with default parameters
    client = GoogleGenAIChatClient()

    # Change model
    # client.set_model("gemini-1.5-pro-latest")

    # Change temperature
    # client.set_temperature(0.7)

    # Change max tokens
    # client.set_max_tokens(256)

    # Invoke the model
    response = client.model.invoke("Tell me a joke.")
    print(response)