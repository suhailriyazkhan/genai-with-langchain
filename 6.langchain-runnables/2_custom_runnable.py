"""
Module for creating and chaining custom Runnable components.

This module defines an abstract Runnable interface and several concrete implementations:
- DummyLLM: Simulates a language model that returns random responses.
- DummyPromptTemplate: Formats prompts using string templates.
- DummyStrOutputParser: Extracts string output from LLM responses.
- RunnableConnector: Chains multiple Runnable instances to form a processing pipeline.

Example usage:
    template1 = DummyPromptTemplate(template='Write a joke about {topic}', input_variables=['topic'])
    template2 = DummyPromptTemplate(template='Explain the following joke {response}', input_variables=['response'])
    llm = DummyLLM()
    parser = DummyStrOutputParser()
    chain = RunnableConnector([template1, llm, template2, llm, parser])
    result = chain.invoke({'topic': 'cricket'})
    print(result)
"""


from abc import ABC, abstractmethod
import random


class Runnable(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass

#Task-based Runnable
class DummyLLM(Runnable):

  def __init__(self):
    print('LLM created')

  def invoke(self, prompt):
    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}


  def predict(self, prompt):

    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}

#Task-based Runnable
class DummyPromptTemplate(Runnable):

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def invoke(self, input_dict):
    return self.template.format(**input_dict)

  def format(self, input_dict):
    return self.template.format(**input_dict)

#Task-based Runnable
class DummyStrOutputParser(Runnable):

  def __init__(self):
    pass

  def invoke(self, input_data):
    return input_data['response']

# Runnable Primitive
# Runnable primitives are the building blocks of a chain.
# The `RunnableConnector` class allows for chaining multiple runnable together.
class RunnableConnector(Runnable):

  def __init__(self, runnable_list):
    self.runnable_list = runnable_list

  def invoke(self, input_data):

    for runnable in self.runnable_list:
      input_data = runnable.invoke(input_data)

    return input_data


# template = DummyPromptTemplate(
#     template='Write a {length} poem about {topic}',
#     input_variables=['length', 'topic']
# )
# llm = DummyLLM()
# parser = DummyStrOutputParser()
# chain = RunnableConnector([template, llm, parser])
# result = chain.invoke({'length': 'short', 'topic': 'rain'})
# print(result)


template1 = DummyPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = DummyPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

llm = DummyLLM()
parser = DummyStrOutputParser()
chain = RunnableConnector([template1, llm, template2, llm, parser])
result = chain.invoke({'topic': 'cricket'})
print(result)