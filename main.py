from huggingface_hub import login
from transformers import AutoModelForCausalLM,  AutoTokenizer
from transformers import pipeline
import torch
import pandas as pd
import json, os

login(token=os.getenv('HF_TOKEN'))
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map=device
)

# generation_pipeline = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer
# )

# input_prompt = [
#     "Hello how are you doing? Not so",
#     "The capital of Canada is"
# ]

# print(generation_pipeline(input_prompt, max_new_tokens=25))

# Tokenizers
# tokenized = tokenizer(input_prompt, padding=True, return_tensors="pt").to(device)

# print(tokenized["input_ids"].shape)
# print(tokenized["input_ids"])
# print(tokenizer.batch_decode(tokenized["input_ids"]))
# print(tokenized["attention_mask"])



# Prompt Templates
# prompt = [
#     {
#         "role": "system",
#         "content": "You are a smart AI assistant who speaks like a pirate."
#     },
#     {
#         "role": "user",
#         "content": "Where does the sun rise?"
#     },
#     {
#         "role": "assistant",
#         "content": "Aye aye"
#     },
# ]

# tokenized = tokenizer.apply_chat_template(
#     prompt,
#     add_generation_prompt=False,
#     continue_final_message=True,
#     tokenize=True,
#     padding=True,
#     return_tensors="pt"
# ).to(device)

# print(tokenized)

# out = model.generate(tokenized, max_new_tokens=32)
# decoded = tokenizer.batch_decode(out)
# print(decoded[0])

df = pd.read_csv("resultset.csv", sep="|")
print(df.head())

# SYSTEM_PROMPT = \
#     {
#         "role": "system",
#         "content": "You are an AI system that reads the date, time, name and message of a group chat text message and imitates the chatting behavior/patterns of only a specific individual, whose name will be provided by the user, in response to a message also provided by the user. No chatting behavior/pattern explanation required."
#     }

# USER_MESSAGE = \
#     {
#         "role": "user",
#         "content": "Name: Aeron\nMessage: Yo dawg, how was chem class today?"
#     }

# POST_MESSAGE = \
#     {
#         "role": "assistant",
#         "content": "Response:"
#     }

# PROMPT = [SYSTEM_PROMPT, USER_MESSAGE, POST_MESSAGE]


# print(SYSTEM_PROMPT["content"])
# print(USER_MESSAGE["content"])
# print(json.dumps(PROMPT, indent=4))

# tokenized = tokenizer.apply_chat_template(
#     PROMPT,
#     continue_final_message=True,
#     padding=True,
#     return_tensors="pt"
# ).to(device)

# out = model.generate(tokenized, max_new_tokens=32)
# decoded = tokenizer.batch_decode(out)
# print(decoded)

# print()
# labels = [d.split("<|start_header_id|>assistant<|end_header_id|>\n\nResponse:")[1].split("<|eot_id|>")[0].strip() for d in decoded]
# for l in labels:
#     print("***")
#     print(l)

# Loss functions
sentence = ["The sky is red"]
tokenized = tokenizer(sentence, return_tensors="pt")["input_ids"]
print(tokenized)
print(tokenizer.batch_decode(tokenized))

input_ids = tokenized[:,:-1]
target_ids = tokenized[:,1:]

print("input seq: ", input_ids)
print("target seq: ", target_ids)
