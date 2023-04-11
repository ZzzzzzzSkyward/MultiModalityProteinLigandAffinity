import json

# 读取 JSON 文件
with open('ligands.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取数组第一个元素
values = set(v[0] for v in data.values())
#去重
values = list(values)

# 将数组第一个元素保存为 txt 文件
with open('pdbid.txt', 'w') as f:
    f.write('\n'.join(values))