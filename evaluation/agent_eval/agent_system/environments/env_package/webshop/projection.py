# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List
import re

def webshop_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        lower_str = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        # Use lowercase for tag detection, but extract from original to preserve
        # case-sensitive WebShop action targets (ASINs, button labels, etc.)
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lower_str.find(start_tag)
        end_idx = lower_str.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = lower_str[-20:]
                continue

            # Extract from original_str to preserve case (WebShop clicks are case-sensitive)
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()

            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = lower_str[-20:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids