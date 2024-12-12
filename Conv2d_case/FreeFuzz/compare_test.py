import os
import torch
import csv
import importlib.util

# 设置包含所有 py 文件的目录路径
#input_dir = "C:\\Users\\14771\\Desktop\\5\\FuMi Major\\baseline\\FreeFuzz-main\\FreeFuzz-main\\src\\config\\output\\cuda-oracle\\success\\torch.nn.functional.conv2d"
input_dir = "./compare-bug/torch.nn.functional.conv2d"
output_csv = "./relative_errors3.csv"

# 初始化 CSV 文件，写入标题
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["file_name", "max_x", "max_y", "relative_error"])  # 包括文件名

# 遍历 py 文件
for py_file in os.listdir(input_dir):
    if py_file.endswith(".py"):
        py_file_path = os.path.join(input_dir, py_file)
        # 动态加载和执行 py 文件
        spec = importlib.util.spec_from_file_location("module.name", py_file_path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # 检查是否有 res_cpu 和 res_gpu
        if "res_cpu" in foo.results and "res_gpu" in foo.results:
            try:
                # 提取 CPU 和 GPU 结果
                x = foo.results["res_cpu"]
                y = foo.results["res_gpu"]
                device = torch.device('cpu')
                x = x.to(device)
                y = y.to(device)
                # 确保是 Tensor 类型并计算相对误差
                if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                    x_flat = x.view(-1)
                    y_flat = y.view(-1)
                    relative_errors = torch.abs(x_flat - y_flat) / torch.abs(x_flat + y_flat)
                    # 找出最大误差
                    max_error_idx = torch.argmax(relative_errors).item()
                    max_error = relative_errors[max_error_idx].item()
                    max_x = x_flat[max_error_idx].item()
                    max_y = y_flat[max_error_idx].item()
                    # 写入 CSV 文件
                    with open(output_csv, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([py_file, max_x, max_y, max_error])
            except Exception as e:
                # 如果结果计算失败，则跳过该文件
                print(f"Error processing file {py_file}: {e}")



