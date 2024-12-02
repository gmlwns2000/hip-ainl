import json
import os

from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
from hip.main.jobs.ga import load_infinite_bench_subset
import transformers
import random


random.seed(2)

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
lines = load_infinite_bench_subset(
    'longbook_qa_eng', tokenizer, 65536 * 3 // 2, 999999,
    add_system_prompt=False, all_answers=True,
)


n_answers = 16

@function
def ask_question(s, question, gt_answers):
    s += system("You are a helpful assistant.")
    s += user(question)
    forks = s.fork(n_answers)
    reveal_answers = "The correct answers include: " + ", ".join(gt_answers) + ". "
    if len(gt_answers) == 1:
        reveal_answers = "The correct answer is: " + gt_answers[0] + ". "
    for i, f in enumerate(forks):
        f += assistant(gen("answer", max_tokens=256, stop_token_ids=[tokenizer.eos_token_id]))
        f += user(
            reveal_answers +
            "Is your answer close (in terms of meaning) to the correct answer? Answer with 'Y' (close) or 'N' (incorrect)."
        )
        f += assistant(gen("is_correct", max_tokens=10, stop_token_ids=[tokenizer.eos_token_id], choices=["Y", "N"], return_logprob=True))
    s += user(
        reveal_answers +
        "Among the candidate answers below, which one is the closest to the correct answer?"
    )
    for i, f in enumerate(forks):
        s += user(f"Candidate Answer #{i + 1}: " + f["answer"] + "; Is Correct? " + f["is_correct"])
    s += user(
        reveal_answers +
        "Among the candidate answers above, which one is the closest to the correct answer? Answer with the number of the correct answer, or 'none' if none of them were correct."
    )
    s += assistant(gen("final", max_tokens=10, stop_token_ids=[tokenizer.eos_token_id],
                       choices=[str(i + 1) for i in range(n_answers)] + ["none"]))


set_default_backend(RuntimeEndpoint("http://localhost:" + os.environ.get("PORT", "33330")))

with open("output.jsonl", "w") as f:
    def run_batch(batch):
        states = ask_question.run_batch(batch, progress_bar=True)

        for state, line in zip(states, batch):
            question = line["question"]
            gt_answers = line["gt_answers"]
            for m in state.messages()[2:]:
                print(m["role"], ":", m["content"])

            print("GT answer:", gt_answers)

            f.write(json.dumps({
                "question": question,
                "gt_answers": gt_answers,
                "answers": [m["content"] for m in state.messages()[3:3+n_answers]],
                "most_correct": state.messages()[-1]["content"],
            }) + "\n")
        f.flush()

    batch = []
    for line in lines:
        question, gt_answers = line
        batch.append({"question": question, "gt_answers": gt_answers})

        if len(batch) < int(os.environ.get("BATCH_SIZE", "1")):
            continue

        run_batch(batch)
        batch = []

    run_batch(batch)

print("Done")
