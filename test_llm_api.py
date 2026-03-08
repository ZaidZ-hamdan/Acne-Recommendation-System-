"""Quick test of Nebius LLM API key and connectivity."""
from openai import OpenAI

LLM_BASE_URL = "https://api.studio.nebius.com/v1/"
LLM_API_KEY = "v1.CmMKHHN0YXRpY2tleS1lMDBma3EycmFrejc5cDNwN2gSIXNlcnZpY2VhY2NvdW50LWUwMHhtbXN3cnpxMHhjcWc0ZDILCK-E68wGEOyJyGc6DAiuh4OYBxCAueaoAUACWgNlMDA.AAAAAAAAAAEPjUVwvLRK9URKwpo45kAv9Zsqt4dlIJMmHnJ35369tyUQIw4tLFOFcoE7iiQQShtRDt2A1wZ18-aNFWGWHbwB"
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def main():
    print("Testing Nebius LLM API (test chat)...\n")
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    system = "You are a helpful assistant for skin and acne. Give short, practical advice. This is not medical advice."
    user_msg = "What's one simple thing I can do for mild acne?"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=256,
        )
        reply = r.choices[0].message.content if r.choices else ""
        print("User:", user_msg)
        print("\nAssistant:", reply)
        print("\n--- Test chat OK ---")
        return 0
    except Exception as e:
        print("Error:", e)
        return 1

if __name__ == "__main__":
    exit(main())
