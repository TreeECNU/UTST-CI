# import os
# device = "cuda:" + str(os.environ.get('LOCAL_RANK',0))
# print(device)
# a = int(os.environ.get("LOCAL_RANK", -1))
# print(a)
import textstat
# import os

# train_file = '../../data/train_prompt_score.json'
# validation_file = '../../data/validation_prompt_score.json'
# if not os.path.exists(train_file):
#     print(f"训练文件不存在: {train_file}")
# if not os.path.exists(validation_file):
#     print(f"验证文件不存在: {validation_file}")

# def get_flesch(text):
#     score = textstat.flesch_reading_ease(text)
#     return score

# text1 = "Neymar and Dani Alves watched basketball with Neymar’s sister Rafaella. Barca beat Real Madrid 85-80 in the Euro League on Thursday night. Real remain top of their division over their bitter rivals by just points difference"
# text2 = "Neymar and Dani Alves watched basketball with Neymar’s sister Rafaella. Barcelona beat Real Madrid 85-80 in the Euro League on Thursday night. Real remain top of their division over their bitter rivals by just points difference."
# text3 = "Neymar and Dani Alves watched basketball with Neymar’s sister Rafaella. Barcelona beat Real Madrid 85-80 in the Euro League on Thursday night. Real remain top of their division over Barcelona by just points difference."
# text4 = "Neymar and Dani Alves watched basketball with Neymar’s sister Rafaella. Barcelona beat Real Madrid 85-80 in the Euro League basketball contest. Real remain top of their division over Barcelona by points difference."

# print(get_flesch(text1))
# print(get_flesch(text2))
# print(get_flesch(text3))
# print(get_flesch(text4))

category_name = "college students"
text = "I am a college student."
input_doc = f"rewrite the following text for {category_name}:\n\n" + text
category = input_doc.split("rewrite the following text for ")[1].split(":\n\n")[0]
doc = input_doc.split(":\n\n")[1]
print(category)
category_name = category.split(" ")[0]
category_name_1 = category.split(" ")[1]
category_name = category_name + " " + category_name_1
print(category_name)
print(doc)