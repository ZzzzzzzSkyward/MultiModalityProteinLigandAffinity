import psutil
import os
import signal
import time
cached={}
def no(proc):
    cpu_times = proc.cpu_times()
    last_time=cached[proc] if proc in cached else 0
    cached[proc]=cpu_times.user
    # 如果该进程在一段时间内没有使用 CPU 时间，认为该进程不再使用 CPU
    ut=cpu_times.user-last_time
    print(ut)
    if ut <1:
        print(f"Process {proc.info['pid']} is not using CPU")
        return True
def kill_matlab():
    # 获取当前所有进程列表
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            # 如果进程名为 MATLAB，且 CPU 占用为 0，则杀死该进程
            if proc.info['name'].find('MATLAB')>=0 and no(proc):
                print(f"Killing MATLAB process {proc.info['pid']}")
                
                os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == '__main__':
    # 定义清理间隔时间（秒）
    interval = 60

    # 循环清理 MATLAB 进程
    while True:
        kill_matlab()
        time.sleep(interval)