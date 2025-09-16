import torch
from torch import nn
from tqdm import tqdm
from transformers import  get_scheduler
from torch.optim import AdamW
from collate_fn import train_dataloader, valid_dataloader, model, id2label, device

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

def test_loop(dataloader, model):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
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




    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    return classification_report(
        true_labels,
        true_predictions,
        mode='strict',
        scheme=IOB2,
        output_dict=True
    )

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
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    metrics = test_loop(valid_dataloader, model)
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        print('saving new weights...\n')
        torch.save(
            model.state_dict(),
            f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
        )
print("Done!")
