import os
import subprocess
import re

# 当前的超参数文件夹名（会用于日志命名）
model_dir_name = "asps-5.22.5"
model_dir = f"workdir/models/{model_dir_name}"

# 测试集路径列表
test_sets = [
    "asps_data/test/CVC-ColonDB",
    "asps_data/test/ETIS-LaribPolypDB",
    "asps_data/test/Kvasir",
    "asps_data/test/CVC-ClinicDB",
    "asps_data/test/EndoScene"
]

# test_sets = [
#     "asps_data/test/CVC-ColonDB",
#     "asps_data/test/ETIS-LaribPolypDB",
# ]
# test_sets = [
#     "breast_data/test"
# ]

# test_sets = [
#     "CAMUS_data_2/test"
# ]

# 正确排序模型文件
# def sort_model_files(files):
#     def extract_epoch(file):
#         match = re.search(r"epoch(\d+)", file)
#         return int(match.group(1)) if match else float('inf')
#     return sorted(files, key=extract_epoch)

# def sort_model_files(files):
#     def extract_keys(file):
#         # 提取 epoch 和 batch 数值（若无 batch 则设为 -1）
#         epoch_match = re.search(r"epoch(\d+)", file)
#         batch_match = re.search(r"batch(\d+)", file)
#         epoch = int(epoch_match.group(1)) if epoch_match else float('inf')
#         batch = int(batch_match.group(1)) if batch_match else -1
#         return (epoch, batch)
    
#     return sorted(files, key=extract_keys)

# 正确排序模型文件：提取前缀数字并按数值升序排列
def sort_model_files(files):
    def extract_step(fname):
        # 假设文件名是 "0000200.pth" 这种格式，去掉扩展名，直接转 int
        base = os.path.splitext(fname)[0]
        try:
            return int(base)
        except ValueError:
            # 如果无法转成 int，就放到最后
            return float('inf')
    return sorted(files, key=extract_step)



# 获取模型文件
model_files = sort_model_files([f for f in os.listdir(model_dir) if f.endswith(".pth")])

# 创建日志输出目录
log_dir = os.path.join("output", f"{model_dir_name}-test")
os.makedirs(log_dir, exist_ok=True)

# 遍历模型和测试集
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"\n=== 测试模型: {model_file} ===")

    for test_path in test_sets:
        test_name = os.path.basename(test_path)
        print(f"-> 测试集: {test_name}")

        # 调用 test.py
        result = subprocess.run(
            [
                "python", "test.py",
                # "python", "test_nolora.py",
                # "--sam_ckpt", model_path,
                "--lora_ckpt", model_path,
                "--data_path", test_path,
                "--run_name", f"eval_{model_file[:-4]}_{test_name}",
                # "--save_pred", "False"
                # "--encoder_adapter",
            ],
            capture_output=True,
            text=True
        )

        # 提取关键结果行
        output_lines = result.stdout.splitlines()
        for line in output_lines[::-1]:  # 从后往前找更快
            if "Test loss:" in line and "metrics:" in line:
                result_line = line
                break
        else:
            result_line = "未找到测试结果。"

        # 保存日志（按测试集分类）
        log_path = os.path.join(log_dir, test_name + ".log")
        with open(log_path, "a") as f:
            f.write(f"{model_file}: {result_line}\n")