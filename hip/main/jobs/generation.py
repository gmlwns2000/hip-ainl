#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models
"""

import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def job_generation(args, model, tokenizer, device):
    parser = argparse.ArgumentParser()

    args.length = 64

    from hip.utils import seed
    seed()

    # Change to your custom prompt text
    with open(args.input, 'r') as f:
        prompt_text = f.read()
    # prompt_text = 'In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance.'
    # prompt_text = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity.'
    print('################## Prompt Text ###################', args.length)
    print(prompt_text, end='||| <-- Here is EOF\n')
    
    input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)

    generate_ids = model.generate(input_ids, max_new_tokens=args.length, min_new_tokens=1)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]
    input_decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
    print("################## Generated Context ###################")
    print(result[len(input_decoded):])