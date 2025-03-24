
from transformers import BertTokenizer

try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("分词器加载成功！")
    print(tokenizer.tokenize("这是一个测试句子。"))
except Exception as e:
    print(f"加载失败，错误信息：{e}")