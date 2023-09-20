import json


def convert_to_dict(file_path):
    """
    读取给定文件并将其转换为包含PDB代码和相关信息的JSON字典

    Args:
        file_path: 文件路径

    Returns:
        包含PDB代码和相关信息的JSON字典
    """
    data = {}
    with open(file_path, "r") as f:
        # 跳过文件头信息
        for line in f:
            if line.startswith("#"):
                continue
            # 解析每行数据
            fields = line.strip().split()
            if fields[-1][0] != "(":
                fields[-2] = fields[-2] + ' ' + fields[-1]
                fields.pop()
            pdb_id = fields[0]
            pdf_file = fields[-2]
            # 删除pdf后缀
            pdf_file = pdf_file[:-4]
            ligand_name = fields[-1]
            try:
                id = ligand_name.rindex("(")
                ligand_name = ligand_name[id:].strip("()")
            except BaseException:
                print(line)
            # 提取PDF文件名和配体名称括号中的字母，并添加到字典中
            letters = "".join(c for c in ligand_name)
            data[pdb_id] = [pdf_file, letters]

    # 将字典转换为JSON格式并返回
    return json.dumps(data)


if __name__ == "__main__":
    # 读取数据并将其转换为JSON格式
    data = convert_to_dict("D:/pdb/index/INDEX_general_PL_data.2020")

    # 将JSON数据保存到文件中
    with open("D:/pdb/ligands_full.json", "w") as f:
        f.write(data)
