import subprocess
import atexit

DEFAULT_ARG_STR = "--height 128 --width 128 --rewardshaping goal"  # --goalmarker hide"  # --cameraangle 90
PROCESS_LIST = []


def launch_k_unity_builds(port_list, build_path, arg_str=None, cwd=None):
    if arg_str is None:
        arg_str = DEFAULT_ARG_STR
    for port in port_list:
        command = [build_path] + arg_str.split() + ["--port", str(port)]
        if cwd is None:
            p = subprocess.Popen(command, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            p = subprocess.Popen(command, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        PROCESS_LIST.append(p)


def cleanup():
    for p in PROCESS_LIST:
        p.kill()
atexit.register(cleanup)
