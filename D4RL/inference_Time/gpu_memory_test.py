import subprocess
import re

def get_gpu_memory_usage():
    # 执行nvidia-smi命令获取GPU状态
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # 解码输出
    output = result.stdout.decode('utf-8')

    # 使用正则表达式解析显存使用情况
    gpu_memory_info = re.findall(r'(\d+),\s*(\d+)', output)

    # 将字符串转换为整数，并以列表形式返回
    gpu_memory_usage = [{'total_memory': int(total), 'used_memory': int(used)} for total, used in gpu_memory_info]

    return gpu_memory_usage

# 使用函数并打印结果
gpu_memory = get_gpu_memory_usage()
for idx, gpu in enumerate(gpu_memory):
    print(f"GPU {idx}: Total Memory: {gpu['total_memory']}B, Used Memory: {gpu['used_memory']} MiB")