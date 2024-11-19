from transformers import TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

from nlp_task.task.text_classification.model import TextClassificationModel
from nlp_task.train.trainer import Trainer

data_path = "./data/data.jsonl"
dataset = Dataset.from_json(data_path)
model = TextClassificationModel()
tokenizer = model.get_tokenizer()


def tokenize_and_align_labels(examples):
    texts = examples["text"]
    targets = examples["target"]
    unique_labels = list(set(targets))  # 去重
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_mapping[i] for i in targets]
    tokenized_inputs = tokenizer.batch_encode_plus(texts)
    tokenized_inputs["labels"] = labels
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

args = TrainingArguments(output_dir="./output", num_train_epochs=3)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)
trainer.train()
