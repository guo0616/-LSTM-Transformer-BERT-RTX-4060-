import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer

# 这里假设你把之前 main.py 里的模型类拿了过来，或者直接从 main.py 导入
from 训练LSTM和BERT模型 import BERT_Classifier, BiLSTM_Classifier

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_VOCAB_SIZE = 30522

print("正在加载分词器和预训练模型权重，请稍候...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 1. 实例化并加载 BERT 模型
bert_model = BERT_Classifier().to(DEVICE)
bert_model.load_state_dict(torch.load('BERT_best.pth', map_location=DEVICE))
bert_model.eval()

# 2. 实例化并加载 BiLSTM 模型 (用于对比)
lstm_model = BiLSTM_Classifier(vocab_size=BERT_VOCAB_SIZE).to(DEVICE)
lstm_model.load_state_dict(torch.load('BiLSTM_best.pth', map_location=DEVICE))
lstm_model.eval()
print("模型加载完毕，服务启动！")


@app.route('/')
def index():
    # 渲染前端页面
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_choice = data.get('model', 'BERT')

    if not text.strip():
        return jsonify({'error': '请输入电影评论文本！'})

    # 数据预处理 (与训练时保持绝对一致)
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    # 禁用梯度计算，加速推理并节省显存
    with torch.no_grad():
        if model_choice == 'BERT':
            outputs = bert_model(input_ids, attention_mask)
        else:
            # 注意：如果你的 LSTM forward 没有用 attention_mask，这里就只传 input_ids
            outputs = lstm_model(input_ids)

            # 将输出转换为概率和预测类别
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    # 0 是 negative，1 是 positive
    sentiment = "Positive (积极 😄)" if pred_class == 1 else "Negative (消极 😡)"

    return jsonify({
        'sentiment': sentiment,
        'confidence': f"{confidence * 100:.2f}%"
    })


if __name__ == '__main__':
    # 开启 debug 模式，并在 5000 端口启动
    app.run(debug=True, host='0.0.0.0', port=5000)