import torch
from torch import nn
from tqdm import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
from collate_fn import train_dataloader, valid_dataloader, test_dataloader, model, id2label, device, tokenizer, test_data
import json
import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)

    model.train()
    for batch_idx, (X, y) in enumerate(dataloader, start=1):
        X = {k: v.to(device) for k, v in X.items()}
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch_idx):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='valid'):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc=f'Evaluating on {mode} set'):
            X = {k: v.to(device) for k, v in X.items()}
            y = y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in labels]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

    report = classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2)
    print(report)
    return classification_report(
        true_labels,
        true_predictions,
        mode='strict',
        scheme=IOB2,
        output_dict=True
    )

def predict_and_save_results(model, tokenizer, test_data, id2label, device, filename='test_data_pred.json'):
    results = []
    print('Predicting labels...')
    model.eval()
    with torch.no_grad():
        for s_idx in tqdm(range(len(test_data))):
            example = test_data[s_idx]
            inputs = tokenizer(example['sentence'], truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pred = model(inputs)
            probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
            predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()

            pred_label = []
            inputs_with_offsets = tokenizer(example['sentence'], return_offsets_mapping=True)
            tokens = inputs_with_offsets.tokens()
            offsets = inputs_with_offsets["offset_mapping"]

            idx = 0
            while idx < len(predictions):
                pred = predictions[idx]
                label = id2label[pred]
                if label != "O":
                    label = label[2:]  # Remove the B- or I-
                    start, end = offsets[idx]
                    all_scores = [probabilities[idx][pred]]
                    # Grab all the tokens labeled with I-label
                    while (
                        idx + 1 < len(predictions) and 
                        id2label[predictions[idx + 1]] == f"I-{label}"
                    ):
                        all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                        _, end = offsets[idx + 1]
                        idx += 1

                    score = np.mean(all_scores).item()
                    word = example['sentence'][start:end]
                    pred_label.append(
                        {
                            "entity_group": label,
                            "score": score,
                            "word": word,
                            "start": start,
                            "end": end,
                        }
                    )
                idx += 1
            results.append(
                {
                    "sentence": example['sentence'], 
                    "pred_label": pred_label, 
                    "true_label": example['labels']
                }
            )
    
    with open(filename, 'wt', encoding='utf-8') as f:
        for example_result in results:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')
    
    print(f'Predictions saved to {filename}')
    return results

# 设置训练
learning_rate = 1e-5
epoch_num = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_f1 = 0.
best_model_path = None

for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    metrics = test_loop(valid_dataloader, model, 'validation')
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
    
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        best_model_path = f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
        print('Saving new weights...\n')
        torch.save(model.state_dict(), best_model_path)

print("Training Done!")

# 加载最佳模型并进行测试集评估
if best_model_path:
    print(f'Loading best model from {best_model_path}')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # 在测试集上评估
    test_metrics = test_loop(test_dataloader, model, 'test')
    
    # 生成预测结果并保存
    predict_and_save_results(model, tokenizer, test_data, id2label, device)
else:
    print("No best model found to evaluate.")
