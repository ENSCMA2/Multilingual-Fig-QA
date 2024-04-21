# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

# login()


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

pipe = transformers.pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	torch_dtype=torch.float16,
	device_map="auto",
)

sequences = pipe(
	['I have tomatoes, basil and cheese at home. What can I cook for dinner?\n'],
	do_sample=True,
	top_k=10,
	num_return_sequences=1,
	eos_token_id=tokenizer.eos_token_id,
	max_length=400,
)

for seq in sequences:
	print(f"{seq['generated_text']}")