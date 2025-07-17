"""
VLA model training module for fine-tuning for action prediction
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
import logging
from typing import List, Dict
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

class WheelLoaderDataset(Dataset):
    """Dataset for wheel loader VQA training"""
    def __init__(self, vqa_pairs: List[Dict], tokenizer, max_length: int = 512):
        self.vqa_pairs = vqa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.vqa_pairs)
    def __getitem__(self, idx):
        pair = self.vqa_pairs[idx]
        #Text format
        input_text = f"Question: {pair['question']}\nContext: {pair['caption']}\nAction:"
        target_text = f"{input_text} {pair['answer']}"
        inputs = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = inputs["input_ids"].clone()
        # input masking for loss computation
        input_only = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length", 
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_length = (input_only["input_ids"] != self.tokenizer.pad_token_id).sum()
        labels[:, :input_length] = -100
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
def train_action_model(data_dir: str, model_dir: str, base_model: str, 
                      batch_size: int = 4, learning_rate: float = 5e-5, epochs: int = 3):
    """Train the action prediction part of VLA""" 
    vqa_file = Path(data_dir) / "vqa_pairs.json"
    if not vqa_file.exists():
        logger.error(f"VQA pairs file not found: {vqa_file}")
        return
    with open(vqa_file, 'r') as f:
        vqa_pairs = json.load(f)
    train_size = int(0.8 * len(vqa_pairs))
    train_pairs = vqa_pairs[:train_size]
    val_pairs = vqa_pairs[train_size:]
    #model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    # token padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    # LORA for PEFT
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # dataset creation
    train_dataset = WheelLoaderDataset(train_pairs, tokenizer)
    val_dataset = WheelLoaderDataset(val_pairs, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=Path(model_dir) / "checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    #Save model
    final_model_path = Path(model_dir) / "final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)