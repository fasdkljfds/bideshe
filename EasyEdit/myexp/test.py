import os

# 如果你使用了 launch.json 中的 MY_WORKSPACE_DIR 环境变量：
workspace_dir = os.environ.get('MY_WORKSPACE_DIR')
print(f"Workspace Directory (from env variable): {workspace_dir}")

# 或者，直接获取当前工作目录：
current_working_dir = os.getcwd()
print(f"Current Working Directory (os.getcwd()): {current_working_dir}")
import sys
print(sys.path)
