import openai

client = openai.OpenAI(api_key='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')

compeletion = client.chat.completions.create(
    model='gpt-4o-mini',
    store = True,
    messages = [
        {'role':'user','content':'hello, how are you?'},
    ]
)

print(compeletion.choices[0].message)