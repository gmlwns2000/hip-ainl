import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import glob
import json

import torch.cuda
import wandb


def main():
    run = wandb.init(
        project="hip-ga-gen-dataset",
    )
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    metadata_json = Path("./outputs/metadata.json")
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    if metadata_json.exists():
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    seen_settings = set()
    settings_pool = []
    latencies = []
    scores = []

    for entry in metadata:
        seen_settings.add(json.dumps(entry["setting"]))

    for pop_file in glob.glob("./saves/pareto/population_gen*.json"):
        with open(pop_file, "r") as f:
            population = json.load(f)
            for setting, score in zip(population['population'], population['scores']):
                setting_json = json.dumps(setting)
                if setting_json in seen_settings:
                    continue
                seen_settings.add(setting_json)
                settings_pool.append(setting)
                latencies.append(score[0])
                scores.append(score[1])

    print(f"Found {len(settings_pool)} unique settings.")
    print(f"Aleady evaluated {len(metadata)} settings.")
    best_score = min(scores)
    print("Best score:", best_score)

    run_idx = 0
    while True:
        run_idx += 1
        # generate 6-digit random id with a-z, A-Z, 0-9
        random_id = ''.join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
        run_id = f"run_{random_id}_{run_idx:03d}"
        print("Run ID:", run_id)

        # Select a random setting, with larger weight for better settings
        cur_setting_idx = random.choices(
            list(range(len(settings_pool))),
            weights=[1 / (score - best_score + 1) for score in scores]
        )[0]
        cur_setting = settings_pool[cur_setting_idx]
        print(f"Selected setting's score: {scores[cur_setting_idx]}")
        print(f"Selected setting's latency: {latencies[cur_setting_idx]}")

        # Write setting to temporary file
        temp_setting_filename = Path(f"./outputs/settings/setting_{run_id}.json")
        temp_setting_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_setting_filename, "w") as f:
            json.dump(cur_setting, f)
        print("Setting written to", temp_setting_filename)

        # Launch SGLang server
        print("Launching SGLang server...")
        print('\n' * 2)
        sglang_process = run_sglang_server(temp_setting_filename.resolve())

        for line in sglang_process.stderr:
            if "The server is fired up and ready to roll" in line.decode("utf-8"):
                break

        def print_output():
            for line in sglang_process.stderr:
                print(line.decode("utf-8"), end="")
        thread = threading.Thread(target=print_output)
        thread.start()

        print("SGLang server is ready.")
        print("Starting evaluation...")

        # Run evaluation
        run_rag = os.environ.get("RUN_RAG", "1") == "1"
        if run_rag:
            rag_result = run_rag_evaluation(f"eval_{run_id}")
            print("RAG evaluation result:", rag_result)

            rag_result_table = wandb.Table(columns=["setting", "rag_results"])
            rag_result_table.add_data(json.dumps(cur_setting), json.dumps(rag_result))
            wandb.log({
                "rag_result": rag_result_table,
            }, step=run_idx, commit=True)

            metadata.append({
                "run_id": run_id,
                "setting": cur_setting,
                "results": [
                    {
                        "task": "rag",
                        "result": rag_result,
                    },
                ],
            })
            write_metadata(metadata, metadata_json, run_id)

        run_ret = os.environ.get("RUN_RET", "1") == "1"
        if run_ret:
            ret_result = run_ret_evaluation(f"eval_{run_id}")
            print("RET evaluation result:", ret_result)

            ret_result_table = wandb.Table(columns=["setting", "ret_results"])
            ret_result_table.add_data(json.dumps(cur_setting), json.dumps(ret_result))
            wandb.log({
                "ret_result": ret_result_table,
            }, step=run_idx, commit=True)

            metadata.append({
                "run_id": run_id,
                "setting": cur_setting,
                "results": [
                    {
                        "task": "rag",
                        "result": ret_result,
                    },
                ],
            })
            write_metadata(metadata, metadata_json, run_id)

        run_ib = os.environ.get("RUN_IB", "1") == "1"
        if run_ib:
            for task in ["choice", "qa", "sum"]:
                ib_result = run_infinibench(f"eval_{run_id}", f"longbook_{task}_eng")
                print(f"Infinibench evaluation result for {task}:", ib_result)

                ib_result_table = wandb.Table(columns=["setting", f"{task}_results"])
                ib_result_table.add_data(json.dumps(cur_setting), ib_result)
                wandb.log({
                    f"ib_{task}_result": ib_result_table,
                }, step=run_idx, commit=True)

                metadata.append({
                    "run_id": run_id,
                    "setting": cur_setting,
                    "results": [
                        {
                            "task": f"infinibench_{task}",
                            "result": ib_result,
                        },
                    ],
                })
                write_metadata(metadata, metadata_json, run_id)

        os.kill(sglang_process.pid, signal.SIGINT)
        print("SGLang server terminated")


def write_metadata(metadata, metadata_json, run_id):
    metadata_json_path = Path(f"./outputs/metadata_{run_id}.json")
    print(f"Writing metadata to {metadata_json_path}...")
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f)

    # Copy metadata to the main metadata file
    with open(metadata_json, "w") as f:
        json.dump(metadata, f)


def run_rag_evaluation(run_name, dataset='nq'):
    try:
        output_path = Path.cwd() / "outputs" / run_name / f"rag_{dataset}"
        print(f"Running RAG generation for {dataset}...")
        print(f"Output path: {output_path}")
        eval_process = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "run_inference.py",
                "--prompt_prefix_path",
                f"./dataset/prompts/rag_128k/rag_{dataset}_128k.txt",
                "--data_dir",
                f"./dataset/data/rag/{dataset}/128k",
                "--split",
                "test",
                "--context_length",
                "128k",
                "--output_path",
                str(output_path / "predictions.jsonl"),
                "--model_url_or_name",
                "openai",
                "--overwrite"
            ],
            cwd="./loft-hip",
            env=os.environ | {
                "IGNORE_PID_MAPPER": "1",
                "ENDPOINT": "http://localhost:33330/v1",
                "BASE_DIR": "./dataset",
                "DATASET": dataset,
            },
        )
        print(f"RAG generation for {dataset} completed. Return code: {eval_process.returncode}")
        eval_process.check_returncode()
        print(f"Evaluating scores for {dataset}...")
        eval_process = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "run_evaluation.py",
                "--answer_file_path",
                f"./dataset/data/rag/{dataset}/128k/test_queries.jsonl",
                "--pred_file_path",
                str(output_path / "predictions.jsonl"),
                "--task_type",
                "rag",
            ],
            cwd="./loft-hip",
            env=os.environ | {
                "BASE_DIR": "./dataset",
                "DATASET": dataset,
            },
        )
        print(f"Evaluation for {dataset} completed. Return code: {eval_process.returncode}")
        eval_process.check_returncode()
        # Read evaluation results
        output_file = output_path / "predictions_metrics.json"
        print(f"Reading evaluation results from {output_file}...")
        with open(output_file, "r") as f:
            metrics = json.load(f)
            return {
                "em": metrics["quality"]["em"],
                "subspan_em": metrics["quality"]["subspan_em"],
                "f1": metrics["quality"]["f1"],
                "num_unanswered_queries": metrics["num_unanswered_queries"],
            }
    except subprocess.CalledProcessError as e:
        print(e)
        return None


def run_ret_evaluation(run_name, dataset='nq'):
    try:
        output_path = Path.cwd() / "outputs" / run_name / f"ret_{dataset}"
        print(f"Running RET generation for {dataset}...")
        print(f"Output path: {output_path}")
        eval_process = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "run_inference.py",
                "--prompt_prefix_path",
                f"./dataset/prompts/retrieval_128k/retrieval_{dataset}_128k.txt",
                "--data_dir",
                f"./dataset/data/retrieval/{dataset}/128k",
                "--split",
                "test",
                "--context_length",
                "128k",
                "--output_path",
                str(output_path / "predictions.jsonl"),
                "--model_url_or_name",
                "openai",
                "--overwrite"
            ],
            cwd="./loft-hip",
            env=os.environ | {
                "IGNORE_PID_MAPPER": "0",
                "ENDPOINT": "http://localhost:33330/v1",
                "BASE_DIR": "./dataset",
                "DATASET": dataset,
            },
        )
        print(f"RET generation for {dataset} completed. Return code: {eval_process.returncode}")
        eval_process.check_returncode()
        print(f"Evaluating scores for {dataset}...")
        eval_process = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "run_evaluation.py",
                "--answer_file_path",
                f"./dataset/data/retrieval/{dataset}/128k/test_queries.jsonl",
                "--pred_file_path",
                str(output_path / "predictions.jsonl"),
                "--task_type",
                "retrieval",
            ],
            cwd="./loft-hip",
            env=os.environ | {
                "BASE_DIR": "./dataset",
                "DATASET": dataset,
            },
        )
        print(f"Evaluation for {dataset} completed. Return code: {eval_process.returncode}")
        eval_process.check_returncode()
        # Read evaluation results
        output_file = output_path / "predictions_metrics.json"
        print(f"Reading evaluation results from {output_file}...")
        with open(output_file, "r") as f:
            metrics = json.load(f)
            return {
                **metrics["quality"],
                "num_unanswered_queries": metrics["num_unanswered_queries"],
            }
    except subprocess.CalledProcessError as e:
        print(e)
        return None


def run_infinibench(run_name, task):
    try:
        output_path = Path.cwd() / "outputs" / run_name / f"infinibench_{task}"
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Running infinibench generation for {task}...")
        print(f"Output path: {output_path}")
        eval_process = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "eval_llama3.py",
                "--task",
                task,
                "--model_path",
                "meta-llama/Llama-3.1-8B-Instruct",
                "--model_name",
                "8b-it-mid",
                "--output_dir",
                str(output_path.resolve())
            ],
            cwd="./InfiniteBench-hip/src",
            env=os.environ | {
                "IS_EXAONE": "0",
                "IS_GEMMA": "0",
                "SGLANG_PORT": "33330",
                "USING_SGLANG": "1",
                "SEQ_LEN": "128",
            },
        )
        print(f"Infinibench generation for {task} completed. Return code: {eval_process.returncode}")
        eval_process.check_returncode()
        print(f"Evaluating scores for {task}...")
        eval_process = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                "compute_scores.py",
                "--task",
                task,
                "--model_name",
                f"llama3-128-8b-it-mid",
                "--output_dir",
                str(output_path.resolve())
            ],
            cwd="./InfiniteBench-hip/src",
            env=os.environ,
            capture_output=True,
        )
        print(f"Evaluation for {task} completed. Return code: {eval_process.returncode}")
        eval_process.check_returncode()
        # Read evaluation results
        result_text = eval_process.stdout.decode("utf-8")
        with open(output_path / "output.txt", "w") as f:
            f.write(result_text)
        return result_text
    except subprocess.CalledProcessError as e:
        print(e)
        return None


def run_sglang_server(population_file):
    return subprocess.Popen(
        [
            str(Path(sys.executable).resolve()),
            "-m",
            "sglang.launch_server",
            "--model-path",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--kv-cache-dtype",
            "auto",
            "--mem-fraction-static",
            "0.6",
            "--tp-size",
            f"{torch.cuda.device_count()}",
            "--chunked-prefill-size",
            "65536",
            "--max-prefill-tokens",
            "65536",
            "--stream-interval",
            "1",
            "--context-length",
            "1200000",
            "--port",
            "33330",
            "--enable-p2p-check",
            "--efficient-weight-load"
        ],
        env=os.environ | {
            "POPULATION_FILE": population_file,
            "SA_BLOCK_SIZE": "64",
            "HIP_PRESET": "mid",
            "HIP_DEBUG_RENDER": "1",
            "HIP_EXTEND_CONTEXT_LENGTH": "131072",
            "DEBUG_NAN": "0",
            "HIP_REFRESH_INTERVAL": "4",
            "SRT_DEBUG_DECODE_SPECIAL_TOKENS": "1",
            "HIP_DEBUG": "0",
            "EXTEND_LEN": "192",
            "HIP_EXTEND": "1",
            "HIP_DISABLE_AUTOTUNE": "1",
            "SRT_ATTENTION_BACKEND": "HIP_ATTN",
            "SRT_MAX_BATCH": "-1",
        },
        #stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    main()
