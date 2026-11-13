import os, json, urllib.request

MODEL = os.environ["OPENAI_MODEL"]
QUESTION = "What is a neural network in one sentence?"


def call_with_sdk():
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from the environment by itself
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
    )
    print(f"SDK response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")


def call_raw_http():
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": QUESTION}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Raw HTTP response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']['prompt_tokens']} in, {result['usage']['completion_tokens']} out")


if __name__ == "__main__":
    print("1. Using the SDK:")
    call_with_sdk()
    print("\n2. Using raw HTTP:")
    call_raw_http()