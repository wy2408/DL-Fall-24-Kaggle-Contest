from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.1, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# download and load competition dataset

from datasets import load_dataset
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", cache_dir="./local_data")
# print and see dataset
prompt = """
You are a great mathematician. You are given a math question along with a proposed solution, and the answer from the solution. 

### Question:
{}

### Solution:
{}

### Answer:
{}

If the answer is right, return 'True', otherwise return 'False'. 

Below is the space for you to give your output.

### Output:
{}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    questions = examples["question"]
    solutions = examples["solution"]
    answers = examples["answer"]
    outputs = examples["is_correct"]
    texts = []

    for question, solution, answer, output in zip(questions, solutions, answers, outputs):
        # Incorporate question, answer, and solution in the prompt
        text = prompt.format(question, solution, answer, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}
# Process the training dataset and generate prompt for each datapoint

train_dataset = dataset['train'].map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 15000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 500,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_8",
        report_to = "none", # Use this for WandB etc
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args
)

model.save_pretrained("lora_model_2")
tokenizer.save_pretrained("lora_model_2")

trainer_stats = trainer.train()
test_dataset = dataset['test']

import pandas as pd

# Prepare a list to store each generated result
generated_results = []
generated_results.append("ID,is_correct")

print('start testing')

for i in range(len(test_dataset['question'])):
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    if i % 100 == 0:
        print(i)
    question = test_dataset['question'][i]
    solution = test_dataset['solution'][i]
    answer = test_dataset['answer'][i]

    # Create the input prompt
    input_prompt = prompt.format(question, solution, answer, "")
    inputs = tokenizer([input_prompt], return_tensors="pt").to("cuda")

    # Run inference to generate output
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode the generated output
    text_generated = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Store result as (ID, is_correct)
    generated_results.append({"ID": i, "is_correct": text_generated[0]})

# Convert results to a DataFrame
df = pd.DataFrame(generated_results)

# Save the results to a CSV file
df.to_csv("output.csv", index=False)