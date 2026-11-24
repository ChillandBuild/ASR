#!/usr/bin/env python3
"""
Evaluates a fine-tuned Whisper Small checkpoint (LoRA adapter or full model)
against the Roman Tanglish test split or individual audio files.
Computes WER and CER metrics and displays predictions.
"""
import sys
import json
import wave
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import tanglish_asr.normalize as norm
import tanglish_asr.evaluate as ev

DEFAULT_CHECKPOINT = ROOT / "private_outputs" / "whisper_small_tanglish_checkpoint" / "best_model"
DEFAULT_TEST_SPLIT = ROOT / "private_outputs" / "training_splits" / "test.jsonl"
DEFAULT_BASE_MODEL = ROOT / "private_outputs" / "hf_prompt_check" / "model"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

def evaluate(args):
    device = get_device()
    print(f"\n==========================================")
    print(f"  Whisper Small Tanglish Model Evaluation")
    print(f"==========================================")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Split: {args.test_split}\n")

    ckpt_path = Path(args.checkpoint)
    base_model_path = Path(args.base_model) if Path(args.base_model).exists() else "openai/whisper-small"

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(str(ckpt_path))

    print("Loading model...")
    is_adapter = (ckpt_path / "adapter_config.json").exists()
    if is_adapter:
        print("Detected LoRA adapter checkpoint. Loading base model + adapter...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.float32
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    else:
        print("Loading full model checkpoint...")
        model = WhisperForConditionalGeneration.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.float32
        )

    model.to(device)
    model.eval()

    with open(args.test_split, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]

    if args.max_samples:
        test_records = test_records[:args.max_samples]

    print(f"Evaluating {len(test_records)} test clips...\n")

    scored_pairs = []
    results = []

    for idx, item in enumerate(test_records):
        cid = item["clip_id"]
        ref = item["text"]
        wav_path = ROOT / item["audio_path"]

        audio = load_audio(wav_path)
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Generate Roman Tanglish output
        prompt_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=prompt_ids,
                max_new_tokens=128,
                no_repeat_ngram_size=4,
                repetition_penalty=1.2
            )
        
        pred_text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        clean_ref = norm.cleanup(ref)
        clean_pred = norm.cleanup(pred_text)
        scored_pairs.append((clean_ref, clean_pred))

        results.append({
            "clip_id": cid,
            "reference": ref,
            "prediction": pred_text,
            "duration_s": item["duration_s"]
        })

        if idx < 5 or (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(test_records)}] {cid} ({item['duration_s']}s):")
            print(f"  Ref:  {ref}")
            print(f"  Pred: {pred_text}")
            print()

    # Metrics
    scores = ev.score_pairs(scored_pairs)
    print(f"==========================================")
    print(f"  Test Evaluation Results ({len(test_records)} clips)")
    print(f"==========================================")
    print(f"Cleaned WER: {scores['wer']*100:.2f}%")
    print(f"Cleaned CER: {scores['cer']*100:.2f}%")
    print(f"==========================================\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_samples": len(test_records),
            "wer": scores["wer"],
            "cer": scores["cer"],
            "checkpoint": args.checkpoint,
            "results": results
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Evaluation report saved to: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper Tanglish Model")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--test_split", type=str, default=str(DEFAULT_TEST_SPLIT))
    parser.add_argument("--base_model", type=str, default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples to evaluate (default: all)")
    parser.add_argument("--language", type=str, default="ta", help="Whisper language prompt id")
    parser.add_argument("--output_json", type=str, default="private_outputs/test_evaluation_results.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
