import json
import os

from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import transformers
import random


def predict_perf(configuration, incomplete_evals, task):
    if task == '':
    @function
    def predict(s):
        s += system(f"You are a performance estimator for machine translation task, where you will estimate the BLEU score for the test architecture.")
