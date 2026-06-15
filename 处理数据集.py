import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer
import os

MAX_LEN = 256


def process_and_save_data():
    csv_path = "IMDB Dataset.csv"
    save_path = "processed_imdb_data"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}，请确保它与此脚本在同一目录下。")

    print("1. 正在读取原始 CSV 文件...")
    df = pd.read_csv(csv_path)

    print("2. 正在转换标签 (positive -> 1, negative -> 0)...")
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # 转换为 Hugging Face Dataset 格式
    dataset = Dataset.from_pandas(df)

    print("3. 正在加载 BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=MAX_LEN)

    print("4. 正在进行 Token 化处理 (利用多核CPU加速，大约需要 1-2 分钟)...")
    # num_proc=4 表示开启 4 个进程加速处理，如果你的 CPU 核心更多也可以改大
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

    print("5. 正在格式化并划分数据集 (80% 训练集, 20% 测试集)...")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataset_dict = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    print(f"6. 正在将处理好的数据保存至 '{save_path}' 文件夹...")
    dataset_dict.save_to_disk(save_path)

    print("✅ 预处理全部完成！你现在可以直接运行 main.py 训练代码了。")


if __name__ == '__main__':
    process_and_save_data()