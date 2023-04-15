import subprocess
import os
import signal
import time
import psutil

def run(command, timeout=2 * 60):
    # 启动子进程
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    # 等待子进程完成，或者超时
    start_time = time.time()
    while psutil.pid_exists(process.pid) and time.time() - start_time < timeout:
        time.sleep(5)
    print(psutil.pid_exists(process.pid), process.pid)
    # 如果子进程未能在超时时间内完成，杀死子进程
    if psutil.pid_exists(process.pid):
        print("kill", process.pid)
        os.kill(process.pid, signal.SIGTERM)
        return False
    else:
        return True

if __name__ == '__main__':
    run("ls")
