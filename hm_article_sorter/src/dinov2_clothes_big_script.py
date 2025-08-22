# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "accelerate>=1.10.0",
#    "datasets>=4.0.0",
#   "evaluate>=0.4.5",
#    "ipykernel>=6.30.1",
#    "pandas>=2.3.1",
#    "pillow>=11.3.0",
#    "pip>=25.2",
#    "pyarrow>=16.0.0",
#    "scikit-learn>=1.7.1",
#   "torch>=2.8.0",
#    "torchao>=0.12.0",
#    "torchvision>=0.23.0",
#    "transformers>=4.55.3",
#   "webdataset>=1.0.2",
# ]
# ///


# %%
from datasets import load_dataset
data_files = {"train": "all_subsets/sampled_train_000000.tar"}

dataset = load_dataset("cubbk/hm_article_sorted_shards_dataset", data_files=data_files)

features_dataset = load_dataset("cubbk/hm_article_sorted_shards_dataset", data_files={"features": "all_subsets/features.json"})

print(dataset)

features = list(features_dataset["features"]["features"][0]) # type: ignore

features

# %%
model_checkpoint = "facebook/dinov2-base" # pre-trained model from which to fine-tune
batch_size = 144 

# %%
# from datasets import load_dataset 

# load a custom dataset from local/remote files or folders using the ImageFolder feature

# option 1: local/remote files
# dataset = load_dataset("jonathan-roberts1/EuroSAT")

# note that you can also provide several splits:
# dataset = load_dataset("imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]})

# note that you can push your dataset to the hub very easily (and reload afterwards using load_dataset)!
# dataset.push_to_hub("nielsr/eurosat")
# dataset.push_to_hub("nielsr/eurosat", private=True)

# option 2: local folder
# dataset = load_dataset("imagefolder", data_dir="path_to_folder")

# option 3: just load any existing dataset from the hub, like CIFAR-10, FashionMNIST ...
# dataset = load_dataset("cifar10")

# %%
import evaluate

metric = evaluate.load("accuracy")



# dataset

# make_smaller = True
# if(make_smaller):
#     # make dataset smaller for quicker training
#     dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))



# %%
example = dataset["train"][1] # type: ignore
example


# %%
dataset["train"].features # type: ignore

# %%
labels = features
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[1]



# %%


from transformers import AutoImageProcessor

image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)
image_processor 



# %%


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = ""
crop_size = ""
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["jpg"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["jpg"]]
    return example_batch



# %%
# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=0.2, shuffle=True) # type: ignore
train_ds = splits['train']
splits_test = splits['test'].train_test_split(test_size=0.5, shuffle=True) # type: ignore

val_ds = splits_test['train']

confirm_last = splits_test['test']

# %%
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# %%
train_ds[0]

# %%


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)



# %%


import os


model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
     f"{model_name}-finetuned-clothes-big",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=True,
    tf32=True,
    optim="adamw_torch_fused",
    torch_compile=True,
    dataloader_num_workers=min(12, (os.cpu_count() or 12)),  # you have 30GB RAM
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_drop_last=True,
    max_grad_norm=0.0,  # disable grad clipping
    # push_to_hub=True,
)



# %%


import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)



# %%
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["cls"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# %%


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor, # type: ignore
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)



# %%
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass



train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()



# %%
metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# %% [markdown]
# INFERENCE

# %%


from PIL import Image
import requests

url = 'https://static.zara.net/assets/public/faa8/0b7b/dc6348e2b336/dc58631e0ded/04695208818-e2/04695208818-e2.jpg?ts=1753797175192&w=1280'
image = Image.open(requests.get(url, stream=True).raw) # type: ignore
image



# %%


from transformers import pipeline

pipe = pipeline("image-classification", "dinov2-base-finetuned-eurosat-clothes")

result = pipe(image)

print(result)



# %%


trainer.push_to_hub()




