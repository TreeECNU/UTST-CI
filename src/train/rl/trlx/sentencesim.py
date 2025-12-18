from sentence_transformers import SentenceTransformer, util
import evaluate
import nltk

# --- 修正1: 下载 nltk 所需的分词模型 ---
# nltk.sent_tokenize 需要 'punkt' 模型，如果第一次运行，需要下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下载 nltk 'punkt' 分词模型...")
    nltk.download('punkt')

# 加载 ROUGE 评估指标
metric_rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    """对预测文本和标签文本进行预处理，用于ROUGE计算"""
    # 确保输入是列表
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(labels, list):
        labels = [labels]

    # 去除首尾空白
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # 将文本按句子分割，并用换行符连接。这是ROUGE评估的标准预处理步骤。
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels

def compute_rouge(predictions, references):
    """
    计算ROUGE分数
    Args:
        predictions: 模型生成的文本列表 (List[str])
        references: 参考标签文本列表 (List[str])
    Returns:
        格式化后的ROUGE分数字典（百分比形式，保留4位小数）
    """
    # 预处理文本
    decoded_preds, decoded_labels = postprocess_text(predictions, references)
    
    # 计算ROUGE分数（使用词干提取器提升匹配效果）
    # --- 修正2: 确保传入的是列表 ---
    rouge_result = metric_rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True  # 启用词干提取，统一词形（如running→run）
    )
    
    # 转换为百分比并格式化
    return {k: round(v * 100, 4) for k, v in rouge_result.items()}

# 加载BGE-M3模型（支持英文，GPU加速）
# 如果没有GPU，请使用 device='cpu'
# model = SentenceTransformer('bge-m3', device='cuda:0')

# ------------------- 示例数据 -------------------
# 示例1：餐厅体验
# pos1 = "The seafood at this restaurant is delicious, the waiters are attentive, and the overall dining experience is enjoyable."
# neg1 = "The seafood at this restaurant is tasteless, the waiters are inattentive, and the overall dining experience is frustrating."

# # 示例2：笔记本电脑
# pos2 = "This laptop has a long-lasting battery, its performance is smooth even for heavy tasks, and using it daily is pleasurable."
# neg2 = "This laptop has a short-lived battery, its performance is laggy even for heavy tasks, and using it daily is annoying."

original_sen = "The scientist did careful tests to get correct results."
trans_1 = "The scientist did careful experiments to obtain precise results."
trans_2 = "The scientist carefully implement the experiments to ensure precise results."


# # --- 语义相似度计算 (BGE-M3) ---
# def calculate_similarity(sentence1, sentence2):
#     """使用BGE-M3计算两个句子的余弦相似度"""
#     emb1 = model.encode(sentence1, normalize_embeddings=True)
#     emb2 = model.encode(sentence2, normalize_embeddings=True)
#     return util.cos_sim(emb1, emb2).item()


# # --- 计算并输出结果 ---
# print("--- BGE-M3 语义相似度 ---")
# score1 = calculate_similarity(original_sen, trans_1)
# score2 = calculate_similarity(original_sen, trans_2)
# print(f"示例1BGE-M3余弦相似度: {score1:.4f}")
# print(f"示例2BGE-M3余弦相似度: {score2:.4f}\n")


print("--- ROUGE 分数 (文本重叠度) ---")
# --- 修正3: 调用时将字符串放入列表中 ---
rouge_score1 = compute_rouge([original_sen], [trans_1])
rouge_score2 = compute_rouge([original_sen], [trans_2])

print("示例1ROUGE分数:", rouge_score1)
print("示例2ROUGE分数:", rouge_score2)