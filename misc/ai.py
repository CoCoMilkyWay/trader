from openai import OpenAI

client = OpenAI(
    api_key="sk-1GTDFWtFdYu0DUb35KLHnyRBjjmQX81XyMQ61fHzmuh5eImj",
    base_url="https://api.feidaapi.com/v1",
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user",
         "content": msg ,
         }
    ]
)

print(completion.choices[0].message.content)

"""
docker run -d -p 3210:3210 ^
-e OPENAI_API_KEY=sk-1GTDFWtFdYu0DUb35KLHnyRBjjmQX81XyMQ61fHzmuh5eImj ^
-e OPENAI_PROXY_URL=https://api.feidaapi.com/v1 ^
--name lobe-chat ^
lobehub/lobe-chat
"""