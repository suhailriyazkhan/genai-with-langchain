import random

class DummyLLM:

  def __init__(self):
    print('LLM created')

  def predict(self, prompt):

    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}

class DummyPromptTemplate:

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def format(self, input_dict):
    return self.template.format(**input_dict)


class DummyLLMChain:

  def __init__(self, llm, prompt):
    self.llm = llm
    self.prompt = prompt

  def run(self, input_dict):

    final_prompt = self.prompt.format(input_dict)
    result = self.llm.predict(final_prompt)

    return result['response']


dummy_template = DummyPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

# Code without chaining
# prompt = dummy_template.format({'length':'short','topic':'india'})
# llm = DummyLLM()
# response = llm.predict(prompt)
# print(response['response'])

# Code with chaining
chain = DummyLLMChain(llm=DummyLLM(), prompt=dummy_template)
print(chain.run({'length':'short', 'topic':'india'}))