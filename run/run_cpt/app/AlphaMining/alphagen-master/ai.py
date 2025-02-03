from openai import OpenAI

client = OpenAI(
  api_key="sk-1GTDFWtFdYu0DUb35KLHnyRBjjmQX81XyMQ61fHzmuh5eImj",
  base_url="https://api.feidaapi.com/v1",
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "yooooo~"}
  ]
)

print(completion.choices[0].message.content)