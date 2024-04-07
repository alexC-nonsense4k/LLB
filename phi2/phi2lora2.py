import random
import math
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
import transformers
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


modelPath = "models--microsoft--phi-2"
tokenizer = AutoTokenizer.from_pretrained(modelPath)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qa_dataset = load_dataset("datasets--rajpurkar--squad_v2")
qa_dataset['train']=qa_dataset['train'].select(range(100))
qa_dataset['validation']=qa_dataset['validation'].select(range(100))
def create_prompt(example):
  if len(example['answers']["text"]) < 1:
    answer = "Cannot Find Answer"
  else:
    answer = example['answers']["text"][0]
  example['prediction'] = f"CONTEXT:\n{example['context']}\n\nQUESTION:\n{example['question']}\n\nANSWER:{answer}"
  return example








mapped_qa_dataset=qa_dataset
mapped_qa_dataset['train']=mapped_qa_dataset['train'].map(create_prompt)
mapped_qa_dataset['validation']=mapped_qa_dataset['validation'].map(create_prompt)


max_len=0
for train in mapped_qa_dataset['train']:
    if  len(tokenizer(train['prediction'])['input_ids'])>max_len:
        max_len=len(tokenizer(train['prediction'])['input_ids'])

mapped_qa_dataset = mapped_qa_dataset.map(lambda samples: tokenizer(samples['prediction'],padding='max_length',max_length=max_len), batched=True)

model = AutoModelForCausalLM.from_pretrained(
    modelPath,
    load_in_8bit=True,
    torch_dtype=torch.float32,
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
        #"k_proj",
    ],
)

epochs=1
peft_model = get_peft_model(model, config)

optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-9)

scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=epochs*len(mapped_qa_dataset['train']))


seed_val = 42
batch_size=2

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device='cuda'

total_t0 = time.time()


for i in range(1):
    print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    peft_model.train()
    step=0
    for j in range(0,len(mapped_qa_dataset['train']),batch_size):
        end=j+batch_size
        if end>=len(mapped_qa_dataset['train']):
            end=len(mapped_qa_dataset['train'])
        train_data=mapped_qa_dataset['train'][j:end]
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(mapped_qa_dataset['train']), elapsed))
        b_input_ids = torch.tensor(train_data['input_ids']).to(device)
        b_input_mask = torch.tensor(train_data['attention_mask']).to(device)
        peft_model.zero_grad()
        result = peft_model(b_input_ids,
                       attention_mask=b_input_mask,
                       return_dict=True)
        logits=result.logits.to('cpu')
        b_input_ids=b_input_ids.to('cpu')
        softmax_tensor = torch.nn.functional.softmax(logits, dim=-1)
        loss_fct = CrossEntropyLoss()
        loss=loss_fct(logits.view(-1,51200),b_input_ids.view(-1))
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step+=1

    # for train_data in mapped_qa_dataset['train']:
    #
    #     if step % 40 == 0 and not step == 0:
    #         elapsed = format_time(time.time() - t0)
    #         print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(mapped_qa_dataset['train']), elapsed))
    #
    #     b_input_ids = torch.tensor(train_data['input_ids']).view(1,-1).to(device)
    #     b_input_mask = torch.tensor(train_data['attention_mask']).view(1,-1).to(device)
    #     peft_model.zero_grad()
    #     result = peft_model(b_input_ids,
    #                    attention_mask=b_input_mask,
    #                    return_dict=True)
    #     logits=result.logits.to('cpu')
    #     b_input_ids=b_input_ids.to('cpu')
    #     softmax_tensor = torch.nn.functional.softmax(logits, dim=-1)
    #     loss_fct = CrossEntropyLoss()
    #     loss=loss_fct(logits.view(-1,51200),b_input_ids.view(-1))
    #     total_train_loss += loss.item()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #     optimizer.step()
    #     scheduler.step()
    #     step+=1
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(mapped_qa_dataset['train'])
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

peft_model.save_pretrained('./phi2-lora')
tokenizer.save_pretrained('./phi2-lora')