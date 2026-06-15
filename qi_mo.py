import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, get_cosine_schedule_with_warmup
from datasets import load_from_disk
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 超参数配置 (适配 RTX 4060 8GB)
# ==========================================
BATCH_SIZE = 16
EPOCHS_LSTM = 10   # LSTM 从头学习，需要更多轮次收敛
EPOCHS_BERT = 3    # BERT 站在巨人的肩膀上，3轮防过拟合
LEARNING_RATE_BERT = 2e-5
LEARNING_RATE_LSTM = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_VOCAB_SIZE = 30522

# ==========================================
# 2. 极速数据加载管道
# ==========================================
def prepare_data():
    print("正在从本地加载已处理好的数据集 'processed_imdb_data'...")
    dataset_dict = load_from_disk("processed_imdb_data")

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    # Windows 下 num_workers=0 避免多进程卡死，pin_memory=True 加速数据锁页传输至 GPU
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    return train_dataloader, eval_dataloader


# ==========================================
# 3. 模型定义
# ==========================================
class BiLSTM_Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super(BiLSTM_Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        # 拼接正向和反向的最后时刻隐状态
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden)


class BERT_Classifier(nn.Module):
    def __init__(self):
        super(BERT_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(self.dropout(pooled_output))

# ==========================================
# 4. 核心训练引擎 (含 AMP 混合精度)
# ==========================================
# 增加了一个传入参数：epochs
def train_and_evaluate(model, model_name, train_dataloader, eval_dataloader, lr, epochs, use_amp=True):
    best_acc = 0.0
    print(f"\n[{model_name}] 开始训练 | 计划轮次: {epochs} | AMP混合精度: {'开启' if use_amp else '关闭'}")
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 这里的总步数改为使用传入的 epochs 计算
    total_steps = len(train_dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=f'runs/{model_name}_amp_{use_amp}')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    global_step = 0
    # 这里的循环范围改为传入的 epochs
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        # 进度条显示也同步更新
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            global_step += 1
            if global_step % 10 == 0:
                writer.add_scalar(f'{model_name}/Training_Loss', loss.item(), global_step)
                writer.add_scalar(f'{model_name}/Learning_Rate', scheduler.get_last_lr()[0], global_step)

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        print(f"[{model_name}] Epoch {epoch + 1} 耗时: {epoch_time:.2f} 秒")

        # --- 验证评估阶段 ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"[{model_name}] Epoch {epoch + 1} 验证集准确率: {acc:.4f}")
        writer.add_scalar(f'{model_name}/Validation_Accuracy', acc, epoch + 1)

    if acc > best_acc:
        best_acc = acc
        # 只有当当前轮次的准确率超越了历史最高记录，才执行保存
        torch.save(model.state_dict(), f'{model_name}_best.pth')
        print(f"🌟 [{model_name}] 发现新的最优模型！已覆盖保存至 {model_name}_best.pth (当前最高: {best_acc:.4f})\n")
    else:
        print(f"[{model_name}] 准确率未提升。(历史最高: {best_acc:.4f})\n")
        # ================================================================

    writer.close()
    print(f"[{model_name}] 训练全部结束！最终可用的最优模型准确率为: {best_acc:.4f}\n")
    # 原来放在循环外面的无脑 save 代码已经删除了

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 1. 瞬间加载数据
    train_dl, eval_dl = prepare_data()

    # 2. 跑 LSTM 模型 (喂给它 10 轮！)
    #lstm_model = BiLSTM_Classifier(vocab_size=BERT_VOCAB_SIZE)
    # 注意这里多传了一个 epochs=EPOCHS_LSTM
    #train_and_evaluate(lstm_model, "BiLSTM", train_dl, eval_dl, lr=LEARNING_RATE_LSTM, epochs=EPOCHS_LSTM, use_amp=True)

    # 释放 LSTM 占用的显存，防止跑 BERT 时 OOM
    #del lstm_model
    #torch.cuda.empty_cache()

    # 3. 跑 BERT 模型 (喂给它 3 轮！)
    bert_model = BERT_Classifier()
    # 注意这里多传了一个 epochs=EPOCHS_BERT
    train_and_evaluate(bert_model, "BERT", train_dl, eval_dl, lr=LEARNING_RATE_BERT, epochs=EPOCHS_BERT, use_amp=True)