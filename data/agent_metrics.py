# import re
# import yaml

# def evaluate_yaml_format(content_str):
#     """
#     判断模型生成的内容是否符合严格的 YAML 约束。
#     返回 1.0 (符合) 或 0.0 (不符合)
#     """
#     # 1. 预处理：移除模型经常乱加的 Markdown 代码块包裹 (```yaml ... ```)
#     # 因为约束要求 YAML 必须在最上方，我们需要提取第一段 --- 包裹的内容
#     clean_str = content_str.strip()
    
#     # 正则逻辑：匹配以 --- 开始，到下一个 --- 结束的最早块
#     # [ \t]* 是为了兼容某些模型在 --- 后面加空格的坏习惯
#     yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*', re.DOTALL)
#     match = yaml_pattern.search(clean_str)
    
#     if not match:
#         return 0.0
    
#     yaml_block = match.group(1)
    
#     try:
#         # 2. 尝试解析 YAML 内容
#         parsed_data = yaml.safe_load(yaml_block)
        
#         # 3. 严格 Key 校验：必须恰好只有 'name' 和 'description'
#         if not isinstance(parsed_data, dict):
#             return 0.0
        
#         required_keys = {'name', 'description'}
#         actual_keys = set(parsed_data.keys())
        
#         # 如果少了 Key，或者多了 Key (比如之前的 'names')，均返回 0.0
#         if actual_keys == required_keys:
#             return 1.0
#         else:
#             return 0.0
            
#     except yaml.YAMLError:
#         # YAML 语法错误（如冒号后没空格、缩进错误等）
#         return 0.0
#     except Exception:
#         return 0.0

# test_case = '''
# ---\nname: heat_and_place_object\ndescription: How to find an item from a storage location, heat it using an appliance, and place it at a target destination.\n---\n\n# Workflow\n1. Locate and Retrieve: Navigate to a typical storage location for the target item (e.g., fridge or cabinet) using go to [storage_location].\n2. Open Storage: If the storage location is closed, open it using open [storage_location].\n3. Take Item: Pick up the item from the storage location using take [item] from [storage_location].\n4. Navigate to Heater: Go to a heating appliance (e.g., microwave) using go to [heating_appliance].\n5. Prepare Heater: If the heating appliance is closed, open it using open [heating_appliance].\n6. Heat Item: Heat the item directly using the appliance with the command heat [item] with [heating_appliance]. (Note: You generally do not need to place the item inside the appliance prior to heating it). \n7. Navigate to Destination: Go to the final target location using go to [destination].\n8. Place Item: Place the heated item at the destination using put [item] in/on [destination] (or equivalent move command).
# '''

# print(evaluate_yaml_format(test_case))
# input()


import json

with open("./alfworld/validation_mapping.jsonl", "r") as f:
    validation_mapping = [json.loads(line.strip()) for line in f][0]

tasks = {
    "pick_and_place":0, # 40
    "pick_two_obj_and_place":0, # 27
    "look_at_obj_in_light":0, # 18
    "pick_heat_then_place_in_recep":0, # 14
    "pick_cool_then_place_in_recep":0, # 24
    "pick_clean_then_place_in_recep":0, # 17
}

validation_res_file = "./math/qwen2.5-7b-7b-skills-alfworld-signal+content0.1+compression0.05/generation/validation/0.jsonl"

with open(validation_res_file, "r") as f:
    validation_results = [json.loads(line.strip()) for line in f]

# print(validation_results[0]['trajectories'][0])

all_tasks = {}
all_tasks_accuracy = {}
for item in validation_results:
    for idx, task in enumerate(item['trajectories']):
        for mapping_task in validation_mapping:
            if mapping_task['observation_text'] in task:
                task_gamefile = mapping_task['extra.gamefile']
                for key in tasks.keys():
                    if key in task_gamefile:
                        task_gamefile = key
                        break
                reward = item['rewards'][idx]
                all_tasks[task_gamefile] = all_tasks.get(task_gamefile, 0) + 1
                all_tasks_accuracy[task_gamefile] = all_tasks_accuracy.get(task_gamefile, 0) + reward/10
                break


print(all_tasks)
print(all_tasks_accuracy)

for task, count in all_tasks.items():
    accuracy = all_tasks_accuracy[task] / count if count > 0 else 0
    print(f"Task: {task}, Count: {count}, Accuracy: {accuracy:.2f}")
# print(tasks)

averaged_accuracy = sum(all_tasks_accuracy.values()) / sum(all_tasks.values()) if sum(all_tasks.values()) > 0 else 0
print(averaged_accuracy)