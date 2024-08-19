# pip3 install ipykernel --upgrade
# python3.11.exe -m ipykernel install --user

import os
import sys
import fire
import time
import glob
import yaml
import shutil
import signal
import logging
import inspect
import functools
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from operator import xor
from pprint import pprint

import qlib
from qlib.config import C
from qlib.workflow import R
from qlib.workflow.cli import render_template
from qlib.utils import set_log_with_config, init_instance_by_config, flatten_dict
from qlib.utils.data import update_config
from qlib.model.trainer import task_train
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.log import get_module_logger
from qlib.tests.data import GetData

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)

# decorator to check the arguments
def only_allow_defined_args(function_to_decorate):
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """Internal wrapper function."""
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if "self" in valid_names:
            valid_names.remove("self")
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError("Unknown argument seen '%s', expected: [%s]" % (arg_name, ", ".join(valid_names)))
        return function_to_decorate(*args, **kwargs)
    return _return_wrapped

# function to handle ctrl z and ctrl c
def handler(signum, frame):
    os.system("kill -9 %d" % os.getpid())

signal.signal(signal.SIGINT, handler)

# function to calculate the mean and std of a list in the results dictionary
def cal_mean_std(results) -> dict:
    mean_std = dict()
    for fn in results:
        mean_std[fn] = dict()
        for metric in results[fn]:
            mean = statistics.mean(results[fn][metric]) if len(results[fn][metric]) > 1 else results[fn][metric][0]
            std = statistics.stdev(results[fn][metric]) if len(results[fn][metric]) > 1 else 0
            mean_std[fn][metric] = [mean, std]
    return mean_std

# function to execute the cmd
def execute_cmd(cmd, wait_when_err=False, raise_err=True):
    print("Running CMD:", cmd)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout: # type: ignore
            sys.stdout.write(line.split("\b")[0])
            if "\b" in line:
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b" * 10 + "\b".join(line.split("\b")[1:-1]))
    if p.returncode != 0:
        if wait_when_err:
            input("Press Enter to Continue")
        if raise_err:
            raise RuntimeError(f"Error when executing command: {cmd}")
        return p.stderr
    else:
        return None

# function to get all the folders benchmark folder
def get_all_folders(models, exclude) -> dict:
    folders = dict()
    if isinstance(models, str):
        model_list = models.split(",")
        models = [m.lower().strip("[ ]") for m in model_list]
    elif isinstance(models, list):
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir("benchmarks")]
    else:
        raise ValueError("Input models type is not supported. Please provide str or list without space.")
    for f in os.scandir("benchmarks"):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path("benchmarks") / f.name
            folders[f.name] = str(path.resolve())
    return folders

# function to get all the files under the model folder
def get_all_files(folder_path, dataset, universe="") -> (str, str): # type: ignore
    if universe != "":
        universe = f"_{universe}"
    yaml_path = str(Path(f"{folder_path}") / f"*{dataset}{universe}.yaml")
    req_path = str(Path(f"{folder_path}") / f"*.txt")
    yaml_file = glob.glob(yaml_path)
    req_file = glob.glob(req_path)
    if len(yaml_file) == 0:
        return None, None
    else:
        return yaml_file[0], req_file[0]

# function to retrieve all the results
def get_all_results(folders) -> dict:
    results = dict()
    for fn in folders:
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            # No experiment results
            continue
        recorders = exp.list_recorders()
        result = dict()
        result["annualized_return_with_cost"] = list()
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        result["ic"] = list()
        result["icir"] = list()
        result["rank_ic"] = list()
        result["rank_icir"] = list()
        for recorder_id in recorders:
            if recorders[recorder_id].status == "FINISHED": # type: ignore
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                if "1day.excess_return_with_cost.annualized_return" not in metrics:
                    print(f"{recorder_id} is skipped due to incomplete result")
                    continue
                result["annualized_return_with_cost"].append(metrics["1day.excess_return_with_cost.annualized_return"])
                result["information_ratio_with_cost"].append(metrics["1day.excess_return_with_cost.information_ratio"])
                result["max_drawdown_with_cost"].append(metrics["1day.excess_return_with_cost.max_drawdown"])
                result["ic"].append(metrics["IC"])
                result["icir"].append(metrics["ICIR"])
                result["rank_ic"].append(metrics["Rank IC"])
                result["rank_icir"].append(metrics["Rank ICIR"])
        results[fn] = result
    return results

# function to generate and save markdown table
def gen_and_save_md_table(metrics, dataset):
    table = "| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for fn in metrics:
        ic = metrics[fn]["ic"]
        icir = metrics[fn]["icir"]
        ric = metrics[fn]["rank_ic"]
        ricir = metrics[fn]["rank_icir"]
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n"
    pprint(table)
    with open("table.md", "w") as f:
        f.write(table)
    return table

# read yaml, remove seed kwargs of model, and then save file in the temp_dir
def gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir):
    with open(yaml_path, "r") as fp:
        config = yaml.safe_load(fp)
    try:
        del config["task"]["model"]["kwargs"]["seed"]
    except KeyError:
        # If the key does not exists, use original yaml
        # NOTE: it is very important if the model most run in original path(when sys.rel_path is used)
        return yaml_path
    else:
        # otherwise, generating a new yaml without random seed
        file_name = yaml_path.split("/")[-1]
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "w") as fp:
            yaml.dump(config, fp)
        return temp_path

class ModelRunner:
    # def __init__(self):
    #     self.run()
        
    def _init_qlib(self, exp_folder_name):
        # init qlib
        provider_uri = "./.qlib/qlib_data/cn_data"
        # config["qlib_init"]["provider_uri"]
        # config["qlib_init"]["region"]
        GetData().qlib_data(
          name="qlib_data",
          target_dir=provider_uri,
          interval="1d",
          region="cn",
          exists_skip=True
          )
        qlib.init(
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(Path(os.getcwd()).resolve() / exp_folder_name),
                    "default_exp_name": "Experiment",
                },
            }
        )

    # function to run the all the models
    @only_allow_defined_args
    def run(
        self,
        models_name=['mlp', 'xgboost'], # None,
        dataset_name="Alpha158",
        universe="",
        exclude=False,
        exp_folder_name: str = "mlruns",
    ):
        """
        models="lightgbm", dataset="Alpha158", universe="csi500" will result in running the following config:
        benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml

        Parameters:
        -----------
        models : str or list
            determines the specific model or list of models to run or exclude.
        exclude : boolean
            determines whether the model being used is excluded or included.
        dataset : str
            determines the dataset to be used for each model.
        universe : str
            the stock universe of the dataset.
            default "" indicates that
        exp_folder_name: str
            the name of the experiment folder

            # Case 7 - run lightgbm model on csi500.
            python run_all_model.py run 3 lightgbm Alpha158 csi500

        """
        self._init_qlib(exp_folder_name)

        # get all folders
        folders = get_all_folders(models_name, exclude)
        # run all the model for iterations
        for idx, fn in enumerate(folders):
            print(fn, folders)
            # get all files
            sys.stderr.write("Retrieving files...\n")
            yaml_path, req_path = get_all_files(folders[fn], dataset_name, universe=universe)
            if yaml_path is None:
                sys.stderr.write(f"There is no {dataset_name}.yaml file in {folders[fn]}")
                continue
            sys.stderr.write("\n")
            
            # Render the template
            rendered_yaml = render_template(yaml_path)
            config = yaml.safe_load(rendered_yaml)
            model = init_instance_by_config(config["task"]["model"])
            dataset = init_instance_by_config(config["task"]["dataset"])
            
            from qlib.data.dataset import DatasetH
            df = dataset.prepare("train")
            print(df)
            
            # start exp recorder R:
            with R.start(experiment_id=str(idx), experiment_name=fn):
                R.log_params(**flatten_dict(config))
                model.fit(dataset)
                R.save_objects(**{"params.pkl": model})
                # prediction
                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()
                # Signal Analysis
                sar = SigAnaRecord(recorder)
                sar.generate()
                # backtest. If users want to use backtest based on their own prediction,
                # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
                par = PortAnaRecord(recorder, config.get("port_analysis_config"), "day")
                par.generate()
            
        self._collect_results(exp_folder_name, dataset_name)

    def _collect_results(self, exp_folder_name, dataset_name):
        folders = get_all_folders(exp_folder_name, dataset_name)
        # getting all results
        sys.stderr.write(f"Retrieving results...\n")
        results = get_all_results(folders)
        if len(results) > 0:
            # calculating the mean and std
            sys.stderr.write(f"Calculating the mean and std of results...\n")
            results = cal_mean_std(results)
            # generating md table
            sys.stderr.write(f"Generating markdown table...\n")
            gen_and_save_md_table(results, dataset_name)
            sys.stderr.write("\n")
        sys.stderr.write("\n")
        # move results folder
        folder_with_stamp = exp_folder_name + f"_{dataset_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        shutil.move(exp_folder_name, folder_with_stamp)
        shutil.move("table.md", f"{folder_with_stamp}/table.md")


if __name__ == "__main__":
    # fire.Fire(ModelRunner)  # run all the model
    runner = ModelRunner()
    runner.run()
