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
from tqdm import tqdm

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
def simplify_transcripts_old(entries, allowed_words, model="gpt-4.1-mini-2025-04-14"):
    simplified = []
    token_counter = 0

    for i, entry in tqdm(enumerate(entries), "Making Prompts", len(entries)):
        text = entry["text"]
        prompt = make_prompt(text, allowed_words)
        try:
            response = client.chat.completions.create(model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7)

            usage = response.usage
            token_counter += usage.total_tokens
            #print(f"Used {usage.total_tokens} tokens this call, {token_counter} total so far.")

            reply = response.choices[0].message.content.strip()
            #print(f"[{i}] ✅")
        except Exception as e:
            print(f"[{i}] ❌ {e}")
            reply = "[ERROR]"

        simplified.append({
            "id": entry["id"],
            "audio_path": entry["file_path"],
            "original_text": text,
            "simplified_text": reply,
            "audio_length_seconds": entry["audio_length_seconds"]
        })
    return simplified

def simplify_transcripts(entries, allowed_words, model="gpt-4.1-mini-2025-04-14"):
    simplified = []
    token_counter = 0

    progress = tqdm(enumerate(entries), desc="Making Prompts", total=len(entries))

    for i, entry in progress:
        text = entry["text"]
        prompt = make_prompt(text, allowed_words)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            usage = response.usage
            token_counter += usage.total_tokens
            reply = response.choices[0].message.content.strip()
            status = "✅"
        except Exception as e:
            reply = "[ERROR]"
            status = f"❌ {str(e).splitlines()[0][:40]}"  # show only brief error

        # Log progress in tqdm
        progress.set_postfix({
            "entry": i,
            "tokens": token_counter,
            "status": status
        })

        simplified.append({
            "id": entry["id"],
            "audio_path": entry["file_path"],
            "original_text": text,
            "simplified_text": reply,
            "audio_length_seconds": entry["audio_length_seconds"]
        })

    return simplified

# 📥 Load LibriSpeech (first 10 for testing)
#ds = load_dataset("librispeech_asr", "clean", split="train.100",trust_remote_code=True)

#

from huggingface_hub import login
login(token=os.environ.get("HUGGINGFACE_API_KEY"))
print("Logged in")


#split = "test"
split = "train"
#split = "validation"

samples_length = 300


from datasets import load_dataset
ds = load_dataset("mozilla-foundation/common_voice_13_0", "en", split=f"{split}[:1%]",trust_remote_code=True)
print("Loaded dataset")

samples = ds.select(range(samples_length))  # start small
print("Selected Samples")


import os
import soundfile as sf

output_dir = "audio_dataset"
os.makedirs(output_dir, exist_ok=True)

metadata = []

for sample in tqdm(samples,"Saving Audio"):

    filename = f"{sample['path']}.wav"
    filepath = os.path.join(output_dir, filename)

    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]
    length_seconds = len(audio_array) / sample_rate

    # Save audio as WAV
    sf.write(filepath, audio_array, sample_rate)

    # Store metadata
    metadata.append({
        "id": sample["path"],
        "file_path": filepath,
        "text": sample["sentence"],
        "audio_length_seconds": length_seconds
    })

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

print("Simplyfying transcripts")

results = simplify_transcripts(metadata, allowed_words)

print(results)

import json
with open(f"{split}_audio_dataset_results_{samples_length}.json", "w") as f:
    json.dump(results, f, indent=2)