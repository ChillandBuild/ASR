#!/usr/bin/env python3
"""
Versioned correction to the original Whisper Small research trainer.

Removes decoder_start_token_id from target labels before Whisper shifts them.
The original trainer remains frozen for reproducibility. Optimization is retained;
current eligible split defaults and a fresh-output guard prevent accidental reuse.
This correction has regression coverage, not a demonstrated accuracy improvement.
Controlled runs must separately freeze their seed, runtime and evaluation recipe.
"""
import os
import sys
import json
import wave
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, PeftModel

# Local paths
DEFAULT_BASE_MODEL = ROOT / "private_outputs" / "hf_prompt_check" / "model"
DEFAULT_PROCESSOR = ROOT / "private_outputs" / "candidate_verification_2026-10-12" / "Dhanush66-rv"
DEFAULT_TRAIN_SPLIT = ROOT / "private_outputs" / "training_splits_owner_audit_2026-10-18" / "train.jsonl"
DEFAULT_VAL_SPLIT = ROOT / "private_outputs" / "training_splits_owner_audit_2026-10-18" / "val.jsonl"
DEFAULT_TEST_SPLIT = ROOT / "private_outputs" / "training_splits_owner_audit_2026-10-18" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "private_outputs" / "whisper_small_tanglish_aligned_2026-10-19"

class TanglishAudioDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: WhisperProcessor, max_samples: int = None):
        self.processor = processor
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        if max_samples:
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        wav_path = ROOT / item["audio_path"]
        
        # Load audio using standard library wave
        with wave.open(str(wav_path), "rb") as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Feature extraction (80-channel log-Mel spectrogram)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        # Target text tokenization
        text = item["text"]
        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "clip_id": item["clip_id"]
        }

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        self.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if self.decoder_start_token_id != 50258:
            raise ValueError("Expected Whisper decoder start token")

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so Cross-Entropy ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Whisper prepends decoder_start_token_id when shifting labels. Its
        # tokenizer's bos_token_id is EOS, so comparing against BOS duplicates SOT.
        if not (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            raise ValueError("Every Whisper target must begin with start-of-transcript")
        labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(args):
    if not args.output_dir.resolve().is_relative_to(ROOT / "private_outputs") or args.output_dir.exists():
        raise ValueError("Use a fresh output directory inside ASR/private_outputs; do not overwrite a checkpoint")
    device = get_device()
    print(f"\n==========================================")
    print(f"  Whisper Small Roman Tanglish Fine-Tuning")
    print(f"==========================================")
    print(f"Device:           {device}")
    print(f"Mode:             {'Full Model Fine-Tuning' if args.full_finetune else 'LoRA Adapter (PEFT)'}")
    print(f"Batch Size:       {args.batch_size} (Grad Accum: {args.grad_accum})")
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Epochs / Steps:   {args.epochs} epochs / max {args.max_steps or 'unlimited'} steps")
    print(f"Output Directory: {args.output_dir}\n")

    # 1. Load Processor & Base Model
    base_model_path = args.base_model if Path(args.base_model).exists() else "openai/whisper-small"
    proc_path = args.processor if Path(args.processor).exists() else "openai/whisper-small"

    print(f"Loading processor from: {proc_path}...")
    processor = WhisperProcessor.from_pretrained(str(proc_path))

    print(f"Loading base model from: {base_model_path}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float32  # standard float32 for MPS / fp16 for CUDA
    )

    # Disable generation config forced decoder tokens to allow Roman Tanglish output
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    # 2. Setup LoRA or Full Fine-Tuning
    if not args.full_finetune:
        print("Configuring LoRA Adapter...")
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Configuring Full Model Fine-Tuning...")
        # Freeze initial convolutional layers for acoustic stability
        model.model.encoder.conv1.requires_grad_(False)
        model.model.encoder.conv2.requires_grad_(False)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

    model.to(device)

    # 3. Datasets & DataLoaders
    print("\nLoading datasets...")
    train_dataset = TanglishAudioDataset(Path(args.train_split), processor, max_samples=args.max_train_samples)
    val_dataset = TanglishAudioDataset(Path(args.val_split), processor, max_samples=args.max_val_samples)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    if args.max_steps and args.max_steps < total_steps:
        total_steps = args.max_steps

    warmup_steps = max(10, int(total_steps * 0.10))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 5. Training Loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    model.train()
    print(f"\n--- Starting Training ({total_steps} optimizer steps) ---")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        step_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            step_loss += loss.item() * args.grad_accum
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    print(f"Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | Loss: {step_loss/args.grad_accum:.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s")
                step_loss = 0.0

                # Periodic Evaluation
                if global_step % args.eval_steps == 0 or global_step == total_steps:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"\n>>> [EVAL] Epoch {epoch+1} Step {global_step} | Val Loss: {val_loss:.4f} <<<")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = args.output_dir / "best_model"
                        print(f"Saving new best checkpoint to: {save_path}...")
                        if not args.full_finetune:
                            model.save_pretrained(str(save_path))
                        else:
                            model.save_pretrained(str(save_path))
                        processor.save_pretrained(str(save_path))
                    print()

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            print("Reached max_steps limit.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Final save
    final_save_path = args.output_dir / "final_model"
    print(f"Saving final model to: {final_save_path}...")
    model.save_pretrained(str(final_save_path))
    processor.save_pretrained(str(final_save_path))
    print("Done!")

def evaluate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_features=input_features, labels=labels)
            total_val_loss += outputs.loss.item()
    model.train()
    return total_val_loss / len(val_loader)

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Small Tanglish Fine-Tuning")
    parser.add_argument("--base_model", type=str, default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--processor", type=str, default=str(DEFAULT_PROCESSOR))
    parser.add_argument("--train_split", type=str, default=str(DEFAULT_TRAIN_SPLIT))
    parser.add_argument("--val_split", type=str, default=str(DEFAULT_VAL_SPLIT))
    parser.add_argument("--test_split", type=str, default=str(DEFAULT_TEST_SPLIT))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full_finetune", action="store_true", help="Perform full fine-tuning instead of LoRA")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=None, help="Limit steps for quick test")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
