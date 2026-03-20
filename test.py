from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="kimi-k2.5",
    api_key="sk-kimi-y7ecstRUdeNiOLpw2iXg9Qvr9H9if2aQxNCf0Z75sLB2EOesTSqVKCrCPM0YXc5L",
    base_url="https://api.kimi.com/coding/v1",
    default_headers={"User-Agent": "claude-code/1.0", "X-Client-Name": "claude-code"},
    temperature=0.3,
)

# Usage is identical to OpenAI
response = llm.invoke([("user", "Hello, how are you?")])
print(response.content)
