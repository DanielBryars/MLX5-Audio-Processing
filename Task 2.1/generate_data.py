import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import json
from words import get_thing_explainer_vocab
from openai import OpenAI

import time
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()


# 🧾 Prompt generator
def make_prompt(original_text, allowed_words_list):
    allowed_str = ", ".join(sorted(allowed_words_list)) + ", ..."
    return (
        f"Rewrite this using only simple words like: {allowed_str}\n"
        f"If a word is not simple, explain it using words from that list.\n\n"
        f"Original: \"{original_text}\"\n"
        f"Thing Explainer Style:"
    )

# 🧠 Simplifier
#
def simplify_transcripts(entries, allowed_words, model="gpt-4"):
    simplified = []
    token_counter = 0

    for i, entry in enumerate(entries):
        text = entry["text"]
        prompt = make_prompt(text, allowed_words)
        try:
            response = client.chat.completions.create(model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7)

            usage = response.usage
            token_counter += usage.total_tokens
            print(f"Used {usage.total_tokens} tokens this call, {token_counter} total so far.")

            reply = response.choices[0].message.content.strip()
            print(f"[{i}] ✅")
        except Exception as e:
            print(f"[{i}] ❌ {e}")
            reply = "[ERROR]"

        simplified.append({
            "id": entry["id"],
            "audio_path": entry["file"],
            "original_text": text,
            "simplified_text": reply
        })
    return simplified




# 📥 Load LibriSpeech (first 10 for testing)
ds = load_dataset("librispeech_asr", "clean", split="train.100",trust_remote_code=True)
samples = ds.select(range(10))  # start small

# 📦 Extract relevant metadata
entries = [{
    "id": s["id"],
    "file": s["file"],
    "text": s["text"]
} for s in samples]

# 🚀 Simplify

#The microscope allows scientists to observe microorganisms
#The microscope allows scientists to observe microorganisms.

# 🔐 Load API key from environment

allowed_words = get_thing_explainer_vocab()

#Explore computer buildings (datacenters), the flat rocks we live on (tectonic plates), the things you use to steer a plane (airliner cockpit controls), and the little bags of water you're made of (cells)


#entries = [{
#    "id": "id",
#    "file": "file",
#    "text": "datacenters"
#}]

results = simplify_transcripts(entries, allowed_words)

# 💾 Save output
with open("simplified_librispeech.json", "w") as f:
    json.dump(results, f, indent=2)


