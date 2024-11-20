from transformers import TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from nlp_task.train.trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from nlp_task.task.text_correction.model import TextCorrectionModel

data_path = "./data/data.jsonl"
ckpt = "/home/dev/model/google-bert/bert-base-chinese/"
dataset = Dataset.from_json(data_path)
model = TextCorrectionModel(ckpt)
tokenizer = model.get_tokenizer()


def tokenize_and_align_labels(examples):
    texts = examples["text"]
    targets = examples["target"]
    texts_inputs = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
    targets_inputs = tokenizer.batch_encode_plus(
        targets, return_tensors="pt", padding=True
    )
    # 纠错 corr_labels
    corr_labels = (texts_inputs["input_ids"] != targets_inputs["input_ids"]).int()
    # print(texts_inputs["input_ids"])
    # print(targets_inputs["input_ids"])
    # print(corr_labels)
    tokenized_inputs = texts_inputs
    tokenized_inputs["corr_labels"] = corr_labels
    tokenized_inputs["labels"] = targets_inputs["input_ids"]
    return tokenized_inputs


tokenized_datasets = dataset.map(
    function=tokenize_and_align_labels,
    batch_size=8,
    batched=True,
    remove_columns=["text", "target"],
    load_from_cache_file=True,
)
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)

args = TrainingArguments(
    output_dir="./output", num_train_epochs=3, save_safetensors=False
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)
trainer.train()
