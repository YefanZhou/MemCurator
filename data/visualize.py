import streamlit as st
import json
import os

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0.0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score


st.set_page_config(layout="wide", page_title="Memory Analysis")

# --- Sidebar ---
st.sidebar.header("📁 Data Source")
base_folder_path = st.sidebar.text_input("Base Folder Path:", value="/home/ouyangsiru/SkillCurator/data/math/qwen2.5-7b-7b-skills+content0.1+compression0.05")
data_type = st.sidebar.radio("Select Data Type:", ('generation', 'rollout'))
training_type = st.sidebar.radio("Select Data Source:", ('training', 'validation'))

folder_path = os.path.join(base_folder_path, data_type, training_type)

def get_jsonl_files(path):
    if not os.path.exists(path):
        return []
    files = [f for f in os.listdir(path) if f.endswith('.jsonl')]
    try:
        # Sort numerically by filename (step number)
        return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        # Fallback to alphabetical sort if filenames are not numbers
        return sorted(files)

files = get_jsonl_files(folder_path)
if not files:
    st.info(f"No .jsonl files found in '{folder_path}'. Please check the path and data type.")
    st.stop()

selected_file = st.sidebar.selectbox("1. Select Step (File):", files)
file_path = os.path.join(folder_path, selected_file)

@st.cache_data
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

all_lines = load_data(file_path)

st.title(f"🔬 Memory Analysis ({data_type.capitalize()})")

if data_type == 'generation':
    line_idx = st.sidebar.number_input(f"2. Select Line (Total {len(all_lines)} lines):", 0, len(all_lines)-1, 0)
    curr_data = all_lines[line_idx]

    is_alfworld = 'trajectories' in curr_data

    if is_alfworld:
        # --- AlfWorld generation log format (generation_alfworld.py) ---
        trajectories = curr_data.get('trajectories', [])
        memories     = curr_data.get('memories', [])
        rewards      = curr_data.get('rewards', [])
        successes    = curr_data.get('successes', [])
        task_idx     = curr_data.get('task_idx', line_idx)

        num_slots = len(trajectories)
        num_success = sum(bool(s) for s in successes)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Task Index", task_idx)
        c2.metric("Success Rate", f"{num_success}/{num_slots}")
        c3.metric("Avg Reward", f"{avg_reward:.2f}")

        st.divider()

        for i in range(num_slots):
            reward_i  = float(rewards[i])  if i < len(rewards)    else 0.0
            success_i = bool(successes[i]) if i < len(successes)   else False
            status_icon = "✅" if success_i else "❌"

            with st.expander(f"Slot {i} | reward={reward_i:.2f} | {status_icon}"):
                col_traj, col_mem = st.columns([3, 2])

                with col_traj:
                    st.caption("Trajectory")
                    traj_text = trajectories[i] if i < len(trajectories) else ""
                    st.code(traj_text, language='text')

                with col_mem:
                    st.caption("Memory / Skills")
                    memory = memories[i] if i < len(memories) else {}
                    skills = memory.get('skills', {}) if isinstance(memory, dict) else {}
                    if skills:
                        st.json(skills)
                    else:
                        st.info("Empty memory")

    else:
        # --- Math QA generation log format ---
        qs = curr_data.get('questions', [])
        golds = curr_data.get('gold_answers', [])
        mems = curr_data.get('memories', [])
        preds = curr_data.get('predictions', [])
        formed_chunks = curr_data.get('formed_chunks', [])

        def clean_text(item):
            if isinstance(item, list) and len(item) > 0:
                return clean_text(item[0])
            return str(item)

        grouped_data = {}

        for i in range(len(qs)):
            q_text = clean_text(qs[i])
            g_text = clean_text(golds[i]) if i < len(golds) else "N/A"

            if q_text not in grouped_data:
                grouped_data[q_text] = {
                    "gold": g_text,
                    "rollouts": []
                }

            p_content = preds[i]
            if isinstance(p_content, list):
                p_content = " ".join([str(x) for x in p_content])

            chunk_content = formed_chunks[i] if i < len(formed_chunks) else "N/A"
            if isinstance(chunk_content, list):
                chunk_content = " ".join([str(x) for x in chunk_content])

            grouped_data[q_text]["rollouts"].append({
                "original_idx": i,
                "memory": mems[i] if i < len(mems) else {},
                "prediction": p_content,
                "chunk": chunk_content
            })

        unique_qs = list(grouped_data.keys())

        if not unique_qs:
            st.warning("No question data found in this line.")
        else:
            rollout_counts = [len(info['rollouts']) for info in grouped_data.values()]
            avg_rollouts = sum(rollout_counts) / len(rollout_counts) if rollout_counts else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", len(qs))
            c2.metric("Unique Questions", len(unique_qs))
            c3.metric("Avg Rollouts per Q", f"{avg_rollouts:.1f}")

            st.divider()

            tabs = st.tabs([f"Q: {q[:25]}..." for q in unique_qs])

            for i, q_text in enumerate(unique_qs):
                with tabs[i]:
                    info = grouped_data[q_text]
                    st.markdown("### ❓ Question")
                    st.info(q_text)
                    st.success(f"**Gold Answer:** {info['gold']}")

                    st.divider()

                    for r in info['rollouts']:
                        is_correct = str(info['gold']).strip().lower() in str(r['prediction']).strip().lower()
                        status_icon = "✅" if is_correct else "❌"

                        with st.expander(f"Original Index: {r['original_idx']} {status_icon}"):
                            st.caption("📄 Formed Chunk (Input Context)")
                            st.code(r['chunk'])

                            st.divider()

                            col_mem, col_pred = st.columns([1, 1])

                            with col_mem:
                                st.caption("🧠 Memory State")
                                memory = r['memory']
                                if isinstance(memory, dict):
                                    if memory.get('core'):
                                        st.markdown("**Core:**")
                                        st.text(memory.get('core', ''))
                                    if memory.get('episodic'):
                                        st.markdown("**Episodic:**")
                                        st.json(memory.get('episodic', {}))
                                    if memory.get('semantic'):
                                        st.markdown("**Semantic:**")
                                        st.json(memory.get('semantic', {}))
                                    if memory.get('skills'):
                                        st.markdown("**Skills:**")
                                        st.json(memory.get('skills', {}))
                                    if not any([memory.get('core'), memory.get('episodic'), memory.get('semantic')]):
                                        st.info("Empty memory")
                                else:
                                    st.json(memory)

                            with col_pred:
                                st.caption("📝 Prediction")
                                if is_correct:
                                    st.success(r['prediction'])
                                else:
                                    st.error(r['prediction'])

elif data_type == 'rollout':
    step_number = os.path.splitext(selected_file)[0]
    st.header(f"Step: {step_number}")
    st.info(f"Displaying {len(all_lines)} items from {selected_file}")

    for i, line_data in enumerate(all_lines):
        input_data = line_data.get('input', 'N/A')
        output_data = line_data.get('output', 'N/A')
        score = line_data.get('score', 'N/A')

        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

        with st.expander(f"Item {i} | Score: {score_str}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📥 Input")
                st.code(str(input_data), language='text')
            with col2:
                st.markdown("##### 📤 Output")
                st.code(str(output_data), language='text')
            
            st.divider()