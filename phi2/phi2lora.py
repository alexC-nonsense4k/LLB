from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
modelPath = "./phi-2"
tokenizer = AutoTokenizer.from_pretrained(modelPath)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qa_dataset = load_dataset("datasets--rajpurkar--squad_v2")
qa_dataset['train']=qa_dataset['train'].select(range(500))
qa_dataset['validation']=qa_dataset['validation'].select(range(200))
def create_prompt(context, question, answer):
  if len(answer["text"]) < 1:
    answer = "Cannot Find Answer"
  else:
    answer = answer["text"][0]
  prompt_template = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}</s>"
  return prompt_template

#applying the reformatting function to the entire dataset
mapped_qa_dataset = qa_dataset.map(lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))
model = AutoModelForCausalLM.from_pretrained(
    modelPath,
    torch_dtype=torch.float16,
    device_map='auto',
)

config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
    ],
)
peft_model = get_peft_model(model, config)
trainable_params = 0
all_param = 0
#iterating over all parameters
for _, param in peft_model.named_parameters():
    #adding parameters to total
    all_param += param.numel()
    #adding parameters to trainable if they require a graident
    if param.requires_grad:
        trainable_params += param.numel()

#printing results
print(f"trainable params: {trainable_params}")
print(f"all params: {all_param}")
print(f"trainable: {100 * trainable_params / all_param:.2f}%")


trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=mapped_qa_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=6,
        learning_rate=1e-3,
        #fp16=True,
        logging_steps=1,
        output_dir='outputs',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()