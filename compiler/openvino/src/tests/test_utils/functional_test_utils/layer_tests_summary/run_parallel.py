# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.conformance_utils import get_logger, progressbar
from utils import constants
from utils import file_utils
from argparse import ArgumentParser
from subprocess import Popen, STDOUT, TimeoutExpired, run, call
from hashlib import sha256
from pathlib import Path
from shutil import rmtree, copyfile
from tarfile import open as tar_open

if not constants.IS_WIN:
    from signal import SIGKILL

import os
import sys
import threading
import csv
import datetime
import shlex

if sys.version_info.major >= 3:
    import _thread as thread
else:
    import thread

has_python_api = True
logger = get_logger('test_parallel_runner')
try:
    from utils.get_available_devices import get_available_devices
except:
    logger.warning("Please set the above env variable to get the same conformance ir names run by run!")
    has_python_api = False

FILENAME_LENGTH = 255
LOG_NAME_REPLACE_STR = "##NAME##"
DEFAULT_PROCESS_TIMEOUT = 3600
DEFAULT_TEST_TIMEOUT = 900
MAX_LENGHT = 4096 if not constants.IS_WIN else 8191

def parse_arguments():
    parser = ArgumentParser()
    exec_file_path_help = "Path to the test executable file"
    cache_path_help = "Path to the cache file with test_name list sorted by execution time. .lst file!"
    worker_num_help = "Worker number. Default value is `cpu_count-1` "
    working_dir_num_help = "Working dir"
    process_timeout_help = "Process timeout in s"
    parallel_help = "Parallel over HW devices. For example run tests over GPU.0, GPU.1 and etc"

    parser.add_argument("-e", "--exec_file", help=exec_file_path_help, type=str, required=True)
    parser.add_argument("-c", "--cache_path", help=cache_path_help, type=str, required=False, default="")
    parser.add_argument("-j", "--workers", help=worker_num_help, type=int, required=False, default=(os.cpu_count() - 1) if os.cpu_count() > 2 else 1)
    parser.add_argument("-p", "--parallel_devices", help=parallel_help, type=int, required=False, default=0)
    parser.add_argument("-w", "--working_dir", help=working_dir_num_help, type=str, required=False, default=".")
    parser.add_argument("-t", "--process_timeout", help=process_timeout_help, type=int, required=False, default=DEFAULT_PROCESS_TIMEOUT)
    return parser.parse_args()

def get_test_command_line_args():
    command_line_args = list()
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--':
            command_line_args = sys.argv[i+1:]
            sys.argv = sys.argv[:i]
            break
    return command_line_args

def get_device_by_args(args: list):
    device = constants.NOT_EXIST_DEVICE
    is_device = False
    for argument in args:
        if "--device" in argument or "-d" == argument[0:2]:
            is_device = True
            if argument.find("=") == -1:
                continue
            device = argument[argument.find("=")+1:]
            break
        if is_device and argument[0] != "-":
            device = argument
            break
    return device

# Class to read test cache    
class TestStructure:
    _name = ""
    _time = 0

    def __init__(self, name, time):
        self._name = name
        self._time = int(time)

class TaskManager:
    process_timeout = -1

    def __init__(self, command_list:list, working_dir: os.path, prev_run_cmd_length=0, device=constants.NOT_EXIST_DEVICE, available_devices=list()):
        self._command_list = command_list
        self._process_list = list()
        self._workers = list()
        self._timers = list()
        self._log_filename = os.path.join(working_dir, f"log_{LOG_NAME_REPLACE_STR}.log")
        self._prev_run_cmd_length = prev_run_cmd_length
        self._idx = 0
        self._device = device
        self._available_devices = [self._device]
        if len(available_devices) > 0:
            self._available_devices = available_devices
        self._device_cnt = len(self._available_devices)

    def __create_thread(self, func):
        thread = threading.Thread(target=func)
        thread.daemon = True
        thread.start()
        return thread

    def init_worker(self):
        if len(self._command_list) <= self._idx:
            logger.warning(f"Skip worker initialiazation. Command list lenght <= worker index")
            return
        if self._device_cnt == 0:
            logger.error(f"Empty available devices! Check your device!")
            exit(-1)
        for target_device in self._available_devices:
            log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx + self._prev_run_cmd_length))
            with open(log_file_name, "w") as log_file:
                args = self._command_list[self._idx].replace(self._device, target_device)
                if not constants.IS_WIN:
                    args = shlex.split(args)
                worker = self.__create_thread(
                    self._process_list.append(Popen(args, shell=constants.IS_WIN, stdout=log_file, stderr=log_file)))
                self._workers.append(worker)
                worker.join()
                self._timers.append(datetime.datetime.now())
                log_file.close()
            # logger.info(f"{self._idx}/{len(self._command_list)} is started")
            self._idx += 1

    @staticmethod
    def kill_process_tree(pid):
        try:
            if not constants.IS_WIN:
                os.killpg(pid, SIGKILL)
            else:
                call(['taskkill', '/F', '/T', '/PID', str(pid)])
        except OSError as err:
            # logger.warning(f"Impossible to kill process {pid} with error: {err}")
            pass

    
    def __find_free_process(self):
        while True:
            for pid in range(len(self._process_list)):
                try:
                    if float((datetime.datetime.now() - self._timers[pid]).total_seconds()) > self.process_timeout:
                        logger.warning(f"Process {pid} exceed time limetattion per process")
                        self.kill_process_tree(self._process_list[pid].pid)
                        self._process_list[pid].kill()
                        self._process_list[pid].wait(timeout=1)
                    self._process_list[pid].wait(timeout=0)
                    args = self._process_list[pid].args
                    if constants.IS_WIN:
                        args = args.split()
                    device = get_device_by_args(args)
                    # logger.info(f"{self._idx}/{len(self._command_list)} is started")
                    return pid, device
                except TimeoutExpired:
                    continue

    def __update_process(self, pid:int, log_file, device):
        args = self._command_list[self._idx].replace(self._device, device)
        if not constants.IS_WIN:
            args = shlex.split(args)
        self._process_list[pid] = Popen(args, shell=constants.IS_WIN, stdout=log_file, stderr=log_file)

    def update_worker(self):
        if self._idx >= len(self._command_list):
            return False
        pid, device = self.__find_free_process()
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx + self._prev_run_cmd_length))
        with open(log_file_name, "w") as log_file:
            self._workers[pid] = self.__create_thread(self.__update_process(pid, log_file, device))
            self._workers[pid].join()
            self._timers[pid] = datetime.datetime.now()
        self._idx += 1
        return True
    
    def compelete_all_processes(self):        
        while len(self._process_list) > 0:
            for pid in range(len(self._process_list)):
                try:
                    if float((datetime.datetime.now() - self._timers[pid]).total_seconds()) > self.process_timeout:
                        logger.warning(f"Process {pid} exceed time limetation per process. The process will be killed")
                        self.kill_process_tree(self._process_list[pid].pid)
                        self._process_list[pid].kill()
                        self._process_list[pid].wait(timeout=1)
                    self._process_list[pid].wait(timeout=0)
                    # logger.info(f"Process {pid} takes {float((datetime.datetime.now() - self._timers[pid]).total_seconds())}")
                    self._process_list.pop(pid)
                    logger.info(f"Compeleting processes: Active process counter: {len(self._process_list)}...")
                    break
                except TimeoutExpired:
                    continue
        return self._idx

class TestParallelRunner:
    def __init__(self, exec_file_path: os.path, test_command_line: list, worker_num: int, working_dir: os.path, cache_path: os.path, is_parallel_devices=False):
        self._exec_file_path = exec_file_path
        self._working_dir = working_dir
        self._conformance_ir_filelists = list()
        self._command = self.__init_basic_command_line_for_exec_file(test_command_line)
        self._worker_num = worker_num
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        if cache_path == "":
            cache_path = os.path.join(self._working_dir, "test_cache.lst")
        self._cache_path = os.path.join(cache_path)
        head, _ = os.path.split(self._cache_path)
        if not os.path.exists(head):
            os.mkdir(head)
        self._is_save_cache = True
        self._disabled_tests = list()
        self._total_test_cnt = 0
        self._device = get_device_by_args(self._command.split())
        self._available_devices = [self._device] if not self._device is None else []
        if has_python_api and is_parallel_devices:
            self._available_devices = get_available_devices(self._device)
            

    def __init_basic_command_line_for_exec_file(self, test_command_line: list):
        command = f'{self._exec_file_path}'
        is_input_folder = False
        for argument in test_command_line:
            if "--input_folders" in argument:
                is_input_folder = True
                command += f" --input_folders="
                argument = argument[argument.find("=")+1:]
            if is_input_folder and argument[0] != "-":
                buf = ""
                for _ in argument.split(','):
                    input_path = argument.replace('"', '')
                    if os.path.isfile(input_path) and file_utils.is_archieve(input_path):
                        input_path = file_utils.unzip_archieve(input_path, self._working_dir)
                    buf = file_utils.prepare_filelist(input_path, ["*.xml"])
                    self._conformance_ir_filelists.append(buf)
                    buf += ","
                argument = buf 
            else:
                is_input_folder = False
                command += f" "
            command += f"{argument}"
        return command

    @staticmethod
    def __replace_restricted_symbols(input_string:str):
        restricted_symbols = "!@$%^&-+`~:;\",<>?"
        for symbol in restricted_symbols:
            input_string = input_string.replace(symbol, '*')
        return input_string

    def __get_test_list_by_runtime(self):
        test_list_file_name = os.path.join(self._working_dir, "test_list.lst")
        if os.path.isfile(test_list_file_name):
            try:
                os.remove(test_list_file_name)
            except Exception as err:
                logger.warning(f"Imposible to remove {test_list_file_name}. Error: {err}")
        command_to_get_test_list = self._command + f' --gtest_list_tests >> {test_list_file_name}'
        logger.info(f"Get test list using command: {command_to_get_test_list}")
        run_res = run(command_to_get_test_list, check=True, shell=True)
        if run_res.stderr != "" and run_res.stderr != None:
            logger.error(f"Ooops! Something is going wrong... {run_res.stderr}")
            exit(-1)

        if not os.path.isfile(test_list_file_name):
            logger.error(f"The test list file does not exists! Please check the process output!")
            exit(-1)

        test_list = list()
        with open(test_list_file_name) as test_list_file:
            test_suite = ""
            for test_name in test_list_file.read().split('\n'):
                pos = test_name.find('#')
                if pos > 0:
                    real_test_name = test_suite + test_name[2:pos-2]
                    if constants.DISABLED_PREFIX in real_test_name:
                        self._disabled_tests.append(real_test_name)
                    else:
                        test_list.append(f'"{self.__replace_restricted_symbols(real_test_name)}":')
                else:
                    test_suite = test_name
            test_list_file.close()
        os.remove(test_list_file_name)
        logger.info(f"Len test_list_runtime (without disabled tests): {len(test_list)}")
        if len(test_list) == 0:
            logger.warning(f"Look like there are not tests to run! Please check the filters!")
            exit(0)
        return test_list

    def __get_test_list_by_cache(self):
        test_list_cache = list()
        if os.path.isfile(self._cache_path):
            logger.info(f"Get test list from cache file: {self._cache_path}")
            with open(self._cache_path, "r") as cache_file:
                for line in cache_file.readlines():
                    pos = line.find(":")
                    time = line[:pos]
                    test_name = line[pos+1:]
                    if not constants.DISABLED_PREFIX in test_name:
                        test_list_cache.append(TestStructure(test_name.replace("\n", ""), time))
        logger.info(f"Len test_list_cache: {len(test_list_cache)}")
        return test_list_cache

    def __generate_test_lists(self, test_list_cache: list, test_list_runtime:list):
        cached_test_list = list()
        runtime_test_test = list()
        cached_test_list_names = list()

        for test in test_list_cache:
            if test._name in test_list_runtime:
                cached_test_list.append(test)
                cached_test_list_names.append(test._name)
        for test in test_list_runtime:
            if not test in cached_test_list_names:
                runtime_test_test.append(test)

        if len(runtime_test_test) > 0:
            logger.warning(f'Cache file is not relevant the run. The will works in hybrid mode.')
            logger.info(f'Test count from cache: {len(cached_test_list)}')
            logger.info(f'Test count from runtime: {len(runtime_test_test)}')
        return cached_test_list, runtime_test_test

    def __prepare_smart_filters(self, proved_test_list:list):
        res_test_filters = list()
        def_length = len(self._command) + len(" --gtest_filter=")
        idx = len(proved_test_list)
        for i in range(len(proved_test_list)):
            if proved_test_list[i]._time == -1:
                idx = i
                break
        
        # Run crashed tests in a separed thread
        if idx < len(proved_test_list):
            while proved_test_list[idx]._time == -1:
                test = proved_test_list.pop(idx)
                res_test_filters.append(test._name)
                if idx >= len(proved_test_list):
                    break

        longest_device = ""
        for device in self._available_devices:
            if len(device) > len(longest_device):
                longest_device = device

        # prepare gtest filters per worker according command line length limitation
        while len(proved_test_list) > 0:
            test_times = []
            is_not_full = True
            worker_test_filters = list()

            real_worker_num = self._worker_num * len(self._available_devices)

            for _ in range(real_worker_num):
                if len(proved_test_list) == 0:
                    break
                worker_test_filters.append(f'"{self.__replace_restricted_symbols(proved_test_list[0]._name)}":')
                test = proved_test_list.pop(0)
                test_times.append(test._time)
            while is_not_full and len(proved_test_list) > 0:
                for i in range(real_worker_num):
                    if i >= len(proved_test_list):
                        break
                    if i == 0:
                        continue
                    while test_times[0] > test_times[i] + proved_test_list[len(proved_test_list) - 1]._time:
                        final_pos = len(proved_test_list) - 1
                        filter = proved_test_list[final_pos]._name
                        if len(worker_test_filters[i]) + def_length + len(filter.replace(self._device, longest_device)) < MAX_LENGHT:
                            worker_test_filters[i] += f'"{self.__replace_restricted_symbols(filter)}":'
                            test_times[i] += proved_test_list[final_pos]._time
                            proved_test_list.pop(final_pos)
                        else:
                            is_not_full = False
                            break
                        if len(proved_test_list) == 0:
                            break
                if is_not_full and len(proved_test_list) > 0:
                    filter = proved_test_list[0]._name
                    if len(worker_test_filters[0]) + def_length + len(filter.replace(self._device, longest_device)) < MAX_LENGHT:
                        worker_test_filters[0] += f'"{self.__replace_restricted_symbols(filter)}":'
                        test_times[0] += proved_test_list[0]._time
                        proved_test_list.pop(0)
                    else:
                        is_not_full = False
            for filter in worker_test_filters:
                res_test_filters.append(filter)
            is_not_full = True
        # logging for debug
        for i in range(len(res_test_filters)):
            filter = res_test_filters[i]
            cnt = filter.count('\":')
            self._total_test_cnt += cnt
            # logger.info(f"Number of tests in job_{i}: {cnt}")
        return res_test_filters
            
    def __get_filters(self):
        if not os.path.isfile(self._exec_file_path):
            logger.error(f"Test executable file {self._exec_file_path} is not exist!")
            sys.exit(-1)
        
        test_list_runtime = self.__get_test_list_by_runtime()
        test_list_cache = self.__get_test_list_by_cache()
        
        cached_test_list, runtime_test_list = self.__generate_test_lists(test_list_cache, test_list_runtime)

        if len(cached_test_list) > 0:
            self._is_save_cache = False
            cached_test_list = self.__prepare_smart_filters(cached_test_list)
        if len(runtime_test_list) > 0:
            self._is_save_cache = True
            self._total_test_cnt += len(runtime_test_list)
            runtime_test_list.reverse()
        logger.info(f"Total test counter is {self._total_test_cnt}")
        return cached_test_list, runtime_test_list
        
    def __execute_tests(self, filters: list(), prev_worker_cnt = 0):
        commands = [f'{self._command} --gtest_filter={filter}' for filter in filters]
        tmp_log_dir = os.path.join(self._working_dir, "temp")
        if not os.path.isdir(tmp_log_dir):
            os.mkdir(tmp_log_dir)
        task_manager = TaskManager(commands, tmp_log_dir, prev_worker_cnt, self._device, self._available_devices)
        for _ in progressbar(range(self._worker_num), "Worker initialization: ", 40):
            task_manager.init_worker()
        for _ in progressbar(range(len(commands) - self._worker_num), "Worker execution: ", 40):
            if not task_manager.update_worker():
                break
        return task_manager.compelete_all_processes()

    def __find_not_runned_tests(self):
        test_names = set()
        interapted_tests = list()
        for log in Path(os.path.join(self._working_dir, "temp")).rglob("log_*.log"):
            log_filename = os.path.join(self._working_dir, log)
            with open(log_filename, "r") as log_file:
                has_status = False
                test_name = None
                try:
                    lines = log_file.readlines()
                except:
                    lines = log.read_text(encoding='ascii', errors='ignore').split('\n')

                for line in lines:
                    if constants.RUN in line:
                        test_name = line[line.find(constants.RUN) + len(constants.RUN) + 1:-1:]
                        has_status = False
                        if test_name is not None:
                            test_names.add(f'"{self.__replace_restricted_symbols(test_name)}":')
                    for _, status_messages in constants.TEST_STATUS.items():
                        for status_msg in status_messages:
                            if status_msg in line:
                                has_status = True
                                break
                        if has_status:
                            break
                if not has_status:
                    interapted_tests.append(f'"{test_name}":')
                log_file.close()
        test_list_runtime = set(self.__get_test_list_by_runtime())
        return list(test_list_runtime.difference(test_names)), interapted_tests

    def run(self):
        if TaskManager.process_timeout == -1:
            TaskManager.process_timeout = DEFAULT_PROCESS_TIMEOUT
        logger.info(f"Run test parallel is started. Worker num is {self._worker_num}")
        if len(self._available_devices) > 1:
            logger.info(f"Tests will be run over devices: {self._available_devices} instead of {self._device}")
        t_start = datetime.datetime.now()
        
        filters_cache, filters_runtime = self.__get_filters()

        worker_cnt = 0
        if len(filters_cache):
            logger.info(f"Execute jobs taken from cache")
            worker_cnt += self.__execute_tests(filters_cache, worker_cnt)
        # 15m for one test in one process
        if TaskManager.process_timeout == -1 or TaskManager.process_timeout == DEFAULT_PROCESS_TIMEOUT:
            TaskManager.process_timeout = DEFAULT_TEST_TIMEOUT
        if len(filters_runtime):
            logger.info(f"Execute jobs taken from runtime")
            worker_cnt += self.__execute_tests(filters_runtime, worker_cnt)
        not_runned_test_filter, interapted_tests = self.__find_not_runned_tests()
        if len(not_runned_test_filter) > 0:
            logger.info(f"Execute not runned {len(not_runned_test_filter)} tests")
            worker_cnt += self.__execute_tests(not_runned_test_filter, worker_cnt)
        if len(interapted_tests) > 0:
            logger.info(f"Execute interapted {len(interapted_tests)} tests")
            worker_cnt += self.__execute_tests(interapted_tests, worker_cnt)

        t_end = datetime.datetime.now()
        total_seconds = (t_end - t_start).total_seconds()
        sec = round(total_seconds % 60, 2)
        min = int(total_seconds / 60) % 60
        h = int(total_seconds / 3600) % 60
        logger.info(f"Run test parallel is finished successfully. Total time is {h}h:{min}m:{sec}s")

    def postprocess_logs(self):
        test_results = dict()
        logger.info(f"Log analize is started")
        saved_tests = list()
        interapted_tests = set()
        INTERAPTED_DIR = "interapted"
        def __save_log(logs_dir, dir, test_name):
            test_log_filename = os.path.join(logs_dir, dir, f"{test_name}.txt".replace('/', '_'))
            hash_str = str(sha256(test_name.encode('utf-8')).hexdigest())
            if hash_str in hash_map.keys():
                (dir_hash, _) = hash_map[hash_str]
                if dir_hash != INTERAPTED_DIR:
                    # logger.warning(f"Test {test_name} was executed before!")
                    return False
            else:
                hash_map.update({hash_str: (dir, test_name)})
            if test_name in interapted_tests:
                if dir == INTERAPTED_DIR:
                    return False
                interapted_log_path = os.path.join(logs_dir, INTERAPTED_DIR, f'{hash_str}.log')
                if os.path.isfile(interapted_log_path):
                    os.remove(interapted_log_path)
                    logger.info(f"LOGS: Interapted {interapted_log_path} will be replaced")
                interapted_tests.remove(test_name)
                hash_map.pop(hash_str)
                hash_map.update({hash_str: (dir, test_name)})
            test_log_filename = os.path.join(logs_dir, dir, f'{hash_str}.log')
            if os.path.isfile(test_log_filename):
                # logger.warning(f"Log file {test_log_filename} is exist!")
                return False
            with open(test_log_filename, "w") as log:
                log.writelines(test_log)
                log.close()
            saved_tests.append(f'\"{test_name}\":')
            return True

        logs_dir = os.path.join(self._working_dir, "logs")
        if os.path.exists(logs_dir):
            logger.info(f"Logs directory {logs_dir} is cleaned up")
            rmtree(logs_dir)
        os.mkdir(logs_dir)
        for test_st, _ in constants.TEST_STATUS.items():
            if not os.path.exists(os.path.join(logs_dir, test_st)):
                os.mkdir(os.path.join(logs_dir, test_st))
        hash_map = dict()
        test_times = list()
        fix_priority = list()
        for log in Path(self._working_dir).rglob("log_*.log"):
            log_filename = os.path.join(self._working_dir, log)
            with open(log_filename, "r") as log_file:
                test_name = None
                test_log = list()
                dir = None
                test_cnt_expected = test_cnt_real_saved_now = 0
                ref_k = None
                try:
                    lines = log_file.readlines()
                except:
                    lines = log.read_text(encoding='ascii', errors='ignore').split('\n')

                for line in lines:
                    if constants.GTEST_FILTER in line:
                        line = line[line.find(constants.GTEST_FILTER):]
                        test_cnt_expected = line.count(':')
                    if constants.RUN in line:
                        test_name = line[line.find(constants.RUN) + len(constants.RUN) + 1:-1:]
                        if self._device != None and self._available_devices != None:
                            for device_name in self._available_devices:
                                if device_name in test_name:
                                    test_name = test_name.replace(device_name, self._device)
                                    break
                    if constants.REF_COEF in line:
                        ref_k = float(line[line.rfind(' ') + 1:])
                    if dir is None:
                        for test_st, mes_list in constants.TEST_STATUS.items():
                            for mes in mes_list:
                                if mes in line:
                                    dir = test_st
                                    break
                            if not dir is None:
                                break
                    # Collect PostgreSQL reporting errors and warnings
                    if (constants.PG_ERR in line) or (constants.PG_WARN in line):
                        test_log.append(line)
                    if test_name is not None:
                        test_log.append(line)
                        if dir:
                            if __save_log(logs_dir, dir, test_name):
                                # update test_cache with tests. If tests is crashed use -1 as unknown time
                                time = -1
                                if "ms)" in line:
                                    time = line[line.rfind("(") + 1:line.rfind("ms)") - 1]
                                test_times.append((int(time), test_name))
                                if dir in test_results.keys():
                                    test_results[dir] += 1
                                else:
                                    test_results[dir] = 1
                                if dir != "passed" and ref_k != None:
                                    fix_priority.append((ref_k, test_name))
                                ref_k = None
                                test_cnt_real_saved_now += 1
                                test_name = None
                                test_log = list()
                                dir = None
                log_file.close()
                if test_name != None:
                    dir = INTERAPTED_DIR
                    if __save_log(logs_dir, dir, test_name):
                        interapted_tests.add(test_name)
                test_cnt_real = test_cnt_real_saved_now
                if test_cnt_real < test_cnt_expected:
                    logger.error(f"Number of tests in {log}: {test_cnt_real}. Expected is {test_cnt_expected} tests")
                else:
                    os.remove(log_filename)
        if len(list(Path(os.path.join(self._working_dir, "temp")).rglob("log_*.log"))) == 0:
            rmtree(os.path.join(self._working_dir, "temp"))
        for test_name in interapted_tests:
            # update test_cache with tests. If tests is crashed use -1 as unknown time
            time = -1
            test_times.append((int(time), test_name))
            if INTERAPTED_DIR in test_results.keys():
                test_results[INTERAPTED_DIR] += 1
            else:
                test_results[INTERAPTED_DIR] = 1
            hash_str = str(sha256(test_name.encode('utf-8')).hexdigest())
            interapted_log_path = os.path.join(logs_dir, INTERAPTED_DIR, f'{hash_str}.log')
            if os.path.isfile(interapted_log_path):
                test_cnt_real_saved_now += 1
        if self._is_save_cache:
            test_times.sort(reverse=True)
            with open(self._cache_path, "w") as cache_file:
                cache_file.writelines([f"{time}:\"" + test_name + "\":\n" for time, test_name in test_times])
                cache_file.close()
                logger.info(f"Test cache test is saved to: {self._cache_path}")
        hash_table_path = os.path.join(logs_dir, "hash_table.csv")
        with open(hash_table_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            csv_writer.writerow(["Dir", "Hash", "Test Name"])
            for hash, st in hash_map.items():
                dir, name = st
                csv_writer.writerow([dir, hash, name])
            logger.info(f"Hashed test list is saved to: {hash_table_path}")
        if len(fix_priority) > 0:
            fix_priority_path = os.path.join(logs_dir, "fix_priority.csv")
            with open(fix_priority_path, "w") as csv_file:
                fix_priority.sort(reverse=True)
                csv_writer = csv.writer(csv_file, dialect='excel')
                csv_writer.writerow(["Test Name", "Fix Priority"])
                ir_hashes = list()
                for priority, name in fix_priority:
                    csv_writer.writerow([name, priority])
                    if "IR=" in name:
                        ir_hashes.append(name[name.find('IR=')+3:name.find('_Device=')])
                logger.info(f"Fix priorities list is saved to: {fix_priority_path}")
                # Find all irs for failed tests
                failed_ir_dir = os.path.join(self._working_dir, f'{self._device}_failed_ir')
                for conformance_ir_filelist in self._conformance_ir_filelists:
                    with open(conformance_ir_filelist, 'r') as file:
                        for conformance_ir in file.readlines():
                            correct_ir = conformance_ir.replace('\n', '')
                            _, tail = os.path.split(correct_ir)
                            ir_hash, _ = os.path.splitext(tail)
                            if ir_hash in ir_hashes:
                                head, _ = os.path.split(conformance_ir_filelist)
                                prefix, _ = os.path.splitext(correct_ir)
                                xml_file = correct_ir
                                bin_file = prefix + constants.BIN_EXTENSION
                                meta_file = prefix + constants.META_EXTENSION
                                
                                failed_ir_xml = xml_file.replace(head, failed_ir_dir)
                                failed_ir_bin = bin_file.replace(head, failed_ir_dir)
                                failed_ir_meta = meta_file.replace(head, failed_ir_dir)

                                dir, _ = os.path.split(failed_ir_xml)
                                if not os.path.isdir(dir):
                                    os.makedirs(dir)
                                copyfile(xml_file, failed_ir_xml)
                                copyfile(bin_file, failed_ir_bin)
                                copyfile(meta_file, failed_ir_meta)
                # api conformance has no failed irs
                if os.path.exists(failed_ir_dir):
                    output_file_name = failed_ir_dir + '.tar'
                    with tar_open(output_file_name, "w:gz") as tar:
                        tar.add(failed_ir_dir, arcname=os.path.basename(failed_ir_dir))
                        logger.info(f"All Conformance IRs for failed tests are saved to: {output_file_name}")
                    rmtree(failed_ir_dir)

        disabled_tests_path = os.path.join(logs_dir, "disabled_tests.log")
        with open(disabled_tests_path, "w") as disabled_tests_file:
            for i in range(len(self._disabled_tests)):
                self._disabled_tests[i] += "\n"
            disabled_tests_file.writelines(self._disabled_tests)
            disabled_tests_file.close()
            logger.info(f"Disabled test list is saved to: {disabled_tests_path}")

        not_run_tests_path = os.path.join(logs_dir, "not_run_tests.log")
        with open(not_run_tests_path, "w") as not_run_tests_path_file:
            test_list_runtime = self.__get_test_list_by_runtime()
            diff_set = set(test_list_runtime).difference(set(saved_tests))
            diff_list = list()
            for item in diff_set:
                diff_list.append(f"{item}\n")
            not_run_tests_path_file.writelines(diff_list)
            not_run_tests_path_file.close()
            logger.info(f"Not run test list is saved to: {not_run_tests_path}")
            l = len(diff_list)
            if l > 0:
                logger.warning(f"Not run test test counter is: {len(diff_list)}")

        is_successfull_run = True
        test_cnt = 0
        for test_st, test_res in test_results.items():
            logger.info(f"{test_st} test counter is: {test_res}")
            test_cnt += test_res
            if (test_st != "passed" and test_st != "skipped") and test_res > 0:
                is_successfull_run = False
        if len(self._disabled_tests):
            logger.info(f"disabled test counter is: {len(self._disabled_tests)}")
        if self._total_test_cnt != test_cnt:
            logger.error(f"Total test count is {test_cnt} is different with expected {self._total_test_cnt} tests")
            is_successfull_run = False
        logger.info(f"Total test count with disabled tests is {test_cnt + len(self._disabled_tests)}. All logs is saved to {logs_dir}")
        return is_successfull_run

if __name__ == "__main__":
    exec_file_args = get_test_command_line_args()
    args = parse_arguments()
    logger.info(f"[ARGUMENTS] --exec_file={args.exec_file}")
    logger.info(f"[ARGUMENTS] --working_dir={args.working_dir}")
    logger.info(f"[ARGUMENTS] --process_timeout={args.process_timeout}")
    logger.info(f"[ARGUMENTS] --cache_path={args.cache_path}")
    logger.info(f"[ARGUMENTS] --workers={args.workers}")
    logger.info(f"[ARGUMENTS] --parallel_devices={args.parallel_devices}")
    logger.info(f"[ARGUMENTS] Executable file arguments = {exec_file_args}")
    TaskManager.process_timeout = args.process_timeout
    conformance = TestParallelRunner(args.exec_file, exec_file_args, args.workers, args.working_dir, args.cache_path, args.parallel_devices)
    conformance.run()
    if not conformance.postprocess_logs():
        logger.error("Run is not successful")
        sys.exit(-1)
    else:
        logger.info("Run is successful")
