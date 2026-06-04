"""
Push文案 × 体育赛事活动用户匹配度分析工具
Gradio界面版
"""
import gradio as gr
import pandas as pd
import openai
import json
import re
import time
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置 ============
BASE_URL = "https://qianfan.baidubce.com/v2/coding"
MODEL = "qianfan-code-latest"
TEST_COUNT = 10  # 快速测试条数

# 评分映射表
TAG_SCORE_MAP = {
    "泛体育运动": 0.9,
    "年轻潮流": 0.7,
    "社交场景": 0.7,
    "夜间经济": 0.6,
    "活动参与型": 0.5,
    "家庭亲子": 0.3,
    "通勤日常": 0.1,
    "美食品质": 0.1,
    "其他": 0.2,
}

THRESHOLD = 0.4

# 全局取消标志
cancel_flag = False

# ============ Prompt模板 ============
SYSTEM_PROMPT = """你是用户行为分析专家。请根据push文案标题，分析该消息最可能吸引什么类型的用户点击。

请从以下标签中选择最匹配的标签（最多3个）：

【核心标签】
- 泛体育运动：文案必须明确提到具体运动项目或运动员（足球、篮球、健身、跑步、马拉松、运动员姓名等）。禁止仅凭"活力""健康""运动"等泛词判定。
- 年轻潮流：文案明确涉及潮牌、球鞋、数码、游戏、潮玩、电竞等年轻人兴趣内容
- 社交场景：文案明确涉及朋友聚餐、聚会、派对、啤酒、夜宵、约饭等多人社交活动。注意："双人餐""家庭餐"不等于社交场景，要看是否有朋友/聚会/派对等社交关键词。
- 夜间经济：文案明确涉及深夜、宵夜、凌晨、晚间特惠等夜间消费场景
- 活动参与型：文案明确涉及薅羊毛、抢券、领券、限时秒杀、打卡、签到、任务等促销活动

【低相关标签】
- 家庭亲子：文案明确涉及亲子、儿童、宝宝、家庭等场景
- 通勤日常：文案明确涉及早餐、午餐、上班族、通勤、工作日等日常刚需场景

【其他】
- 如果以上标签都不匹配，返回 ["其他"]

严格判断标准：
1. 必须基于文案实际内容判断，不要臆想或联想
2. "双人餐""超值套餐"等餐饮优惠，如果没有明确社交关键词（聚会、派对、朋友、约饭），不要归为社交场景
3. 领券、优惠、特惠等促销内容，归为"活动参与型"
4. 无法确定时，返回 ["其他"]，不要强行匹配

请以JSON数组格式返回，每条包含：
- "id": 序号（必须与输入序号一致，从0开始）
- "tags": ["标签1", "标签2"] 或 ["其他"]
- "reason": "简短理由，必须引用文案中的关键词"

重要：数组顺序必须与输入顺序完全一致，id从0开始连续递增。不要打乱顺序。
只返回JSON，不要其他内容。"""


def call_llm_with_retry(client, model, messages, max_tokens=4000, max_retries=5):
    """带重试的LLM调用"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return resp
        except openai.RateLimitError:
            wait_time = 15 * (attempt + 1)
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(5)
    raise Exception("API调用失败，重试次数用尽")


def extract_json_from_text(text):
    """从文本中提取JSON数组，兼容模型返回额外内容的情况"""
    candidates = [text]
    code_block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if code_block:
        candidates.append(code_block.group(1))
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        candidates.append(array_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def analyze_batch(titles, client, model):
    """分析一批文案标题，保证返回数量与输入一致"""
    titles_text = "\n".join([f"{i}. {title}" for i, title in enumerate(titles)])
    prompt = f"{SYSTEM_PROMPT}\n\n以下是待分析的文案：\n{titles_text}"

    resp = call_llm_with_retry(
        client, model,
        [{"role": "user", "content": prompt}],
        max_tokens=4000
    )

    raw = resp.choices[0].message.content.strip()
    result = extract_json_from_text(raw)

    if result is None:
        print(f"[DEBUG] LLM返回内容无法解析为JSON: {raw[:500]}...")
        raise ValueError(f"LLM返回内容不是有效JSON: {raw[:100]}")

    # 用id字段校验顺序，防止LLM乱序返回
    expected = len(titles)
    if len(result) == expected and all("id" in item for item in result):
        try:
            # 按id字段重新排序
            result.sort(key=lambda x: int(x["id"]))
        except (ValueError, KeyError):
            print(f"[WARNING] id字段排序失败，使用原始顺序")

    # 保证返回数量与输入一致
    if len(result) != expected:
        print(f"[WARNING] 数量不匹配：输入 {expected} 条，返回 {len(result)} 条")
        default = {"tags": ["其他"], "reason": "LLM返回数量不足"}
        result = (result + [default] * expected)[:expected]

    return result


def calculate_score(tags):
    """根据标签计算匹配度分数"""
    if not tags:
        return 0.0
    return max((TAG_SCORE_MAP.get(tag, 0.2) for tag in tags), default=0.0)


def get_sheet_names(file):
    """获取Excel文件的所有sheet名"""
    if file is None:
        return []
    xls = pd.ExcelFile(file.name)
    return xls.sheet_names


def cancel_processing():
    """取消处理"""
    global cancel_flag
    cancel_flag = True
    return "正在取消..."


def _build_report(total, all_results, all_tags_counter, cancel_flag, df=None, title_col=None, test_mode=False):
    """生成分析报告"""
    if test_mode:
        # 测试模式：逐行表格
        report = f"## 快速测试结果（前{TEST_COUNT}条）\n\n"
        report += "| 序号 | 文案标题 | 匹配 | 分数 | 标签 | 理由 |\n"
        report += "|------|----------|------|------|------|------|\n"
        for i, row in df.iterrows():
            title = str(row[title_col])
            title_short = title[:30] + ("..." if len(title) > 30 else "")
            match_icon = "✅" if row["人群匹配"] == 1 else "❌"
            report += f"| {i+1} | {title_short} | {match_icon} | {row['匹配度分数']} | {row['匹配标签']} | {row['匹配理由']} |\n"
        return report

    # 全量模式：统计报告
    bool_counts = Counter(df["人群匹配"])
    score_series = df["匹配度分数"]
    score_dist = pd.cut(
        score_series,
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1.01],
        labels=["0.0-0.2", "0.3-0.4", "0.5-0.6", "0.7-0.8", "0.9-1.0"],
        right=False
    ).value_counts().to_dict()

    status = "已取消" if cancel_flag else "分析完成"
    report = f"""
## {status}

### 基本统计
- 总条数：{total}
- 匹配（=1）：{bool_counts.get(1, 0)} 条 ({bool_counts.get(1, 0)/total*100:.1f}%)
- 不匹配（=0）：{bool_counts.get(0, 0)} 条 ({bool_counts.get(0, 0)/total*100:.1f}%)

### 匹配度分数分布
| 分数段 | 数量 | 占比 |
|--------|------|------|
| 0.9-1.0 | {score_dist.get("0.9-1.0", 0)} | {score_dist.get("0.9-1.0", 0)/total*100:.1f}% |
| 0.7-0.8 | {score_dist.get("0.7-0.8", 0)} | {score_dist.get("0.7-0.8", 0)/total*100:.1f}% |
| 0.5-0.6 | {score_dist.get("0.5-0.6", 0)} | {score_dist.get("0.5-0.6", 0)/total*100:.1f}% |
| 0.3-0.4 | {score_dist.get("0.3-0.4", 0)} | {score_dist.get("0.3-0.4", 0)/total*100:.1f}% |
| 0.0-0.2 | {score_dist.get("0.0-0.2", 0)} | {score_dist.get("0.0-0.2", 0)/total*100:.1f}% |

### 用户标签频次Top10
"""
    for tag, count in all_tags_counter.most_common(10):
        report += f"- {tag}: {count}次\n"

    return report


def process_excel(file, sheet_name, title_col_idx, api_key, base_url, model, batch_size, concurrency=3, test_mode=False, progress=gr.Progress()):
    """处理Excel文件。支持并行处理加速。test_mode=True 时只处理前 TEST_COUNT 条。"""
    global cancel_flag
    cancel_flag = False

    if not api_key:
        raise gr.Error("请填写API Key")
    if file is None:
        raise gr.Error("请上传文件")

    # 读取指定sheet
    df = pd.read_excel(file.name, sheet_name=sheet_name)

    # 列号从1开始，转换为从0开始的索引
    title_col_idx = int(title_col_idx) - 1
    if title_col_idx < 0 or title_col_idx >= len(df.columns):
        raise gr.Error(f"列号无效，该Sheet共有{len(df.columns)}列（1-{len(df.columns)}）")

    # 测试模式截取前N条
    if test_mode:
        df = df.head(TEST_COUNT).copy()

    # 获取文案标题列
    title_col = df.columns[title_col_idx]
    titles = df[title_col].fillna("").astype(str).tolist()
    total = len(titles)

    # 初始化LLM客户端
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # 分批处理
    batch_size = int(batch_size)
    concurrency = int(concurrency)
    batches = []
    for batch_idx in range(0, total, batch_size):
        batch_titles = titles[batch_idx:batch_idx + batch_size]
        batches.append((batch_idx // batch_size, batch_titles))

    total_batches = len(batches)
    completed_batches = 0
    batch_results_map = {}  # {batch_num: results}

    progress(0, desc=f"开始{'测试' if test_mode else '分析'}... 共{total_batches}批，并发{concurrency}")

    def process_single_batch(batch_info):
        """处理单个批次（在线程中执行）"""
        batch_num, batch_titles = batch_info
        try:
            results = analyze_batch(batch_titles, client, model)
            processed = []
            for item in results:
                tags = item.get("tags", [])
                reason = item.get("reason", "")
                score = calculate_score(tags)
                bool_val = 1 if score >= THRESHOLD else 0
                processed.append({"bool": bool_val, "score": round(score, 2), "tags": tags, "reason": reason})
            return batch_num, processed, None
        except Exception as e:
            error_result = [{"bool": 0, "score": 0.0, "tags": ["分析失败"], "reason": str(e)[:200]}] * len(batch_titles)
            return batch_num, error_result, str(e)

    # 并行处理
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_single_batch, batch): batch[0] for batch in batches}

        for future in as_completed(futures):
            if cancel_flag:
                # 取消时停止提交新任务
                for f in futures:
                    f.cancel()
                break

            batch_num, results, error = future.result()
            batch_results_map[batch_num] = results
            completed_batches += 1

            if error:
                print(f"第 {batch_num + 1} 批处理失败: {error}")

            progress(completed_batches / total_batches, desc=f"已完成 {completed_batches}/{total_batches} 批...")

    # 按批次顺序组装结果
    all_results = []
    all_tags_counter = Counter()
    for i in range(total_batches):
        if i in batch_results_map:
            for item in batch_results_map[i]:
                all_results.append(item)
                all_tags_counter.update(item.get("tags", []))

    # 取消时填充剩余
    if cancel_flag:
        remaining = total - len(all_results)
        all_results.extend([{"bool": 0, "score": 0.0, "tags": ["已取消"], "reason": "用户取消"}] * remaining)

    # 确保结果数量匹配
    all_results.extend([{"bool": 0, "score": 0.0, "tags": ["未处理"], "reason": ""}] * (total - len(all_results)))

    # 写入DataFrame
    df["人群匹配"] = [r["bool"] for r in all_results[:total]]
    df["匹配度分数"] = [r["score"] for r in all_results[:total]]
    df["匹配标签"] = [", ".join(r["tags"]) for r in all_results[:total]]
    df["匹配理由"] = [r["reason"] for r in all_results[:total]]

    # 保存结果文件
    output_path = None
    if not test_mode:
        output_path = os.path.join(os.path.dirname(file.name), "匹配分析结果.xlsx")
        df.to_excel(output_path, index=False)

    # 生成报告
    report = _build_report(total, all_results, all_tags_counter, cancel_flag, df, title_col, test_mode)

    progress(1.0, desc="完成！")
    return output_path, report


# ============ Gradio界面 ============
with gr.Blocks(title="Push文案用户匹配分析", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📊 Push文案 × 体育赛事活动 用户匹配度分析
        上传Excel文件，自动分析每条文案与大型体育赛事活动目标用户的兴趣匹配度
        """
    )

    with gr.Row():
        # 左侧：文件和配置
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📁 文件配置")
                file_input = gr.File(label="上传Excel文件", file_types=[".xlsx", ".xls"])

                with gr.Row():
                    sheet_dropdown = gr.Dropdown(label="选择Sheet", choices=[], interactive=True)
                    refresh_btn = gr.Button("🔄 刷新", size="sm")

                title_col_idx = gr.Number(label="文案标题所在列（从1开始）", value=4, precision=1)

            with gr.Group():
                gr.Markdown("### ⚙️ API配置")
                api_key = gr.Textbox(label="API Key", placeholder="请输入API Key", type="password")
                base_url = gr.Textbox(label="Base URL", value=BASE_URL)
                model = gr.Textbox(label="模型名称", value=MODEL)

            with gr.Group():
                gr.Markdown("### 📋 处理参数")
                batch_size = gr.Slider(
                    label="每批处理条数",
                    minimum=20,
                    maximum=200,
                    value=50,
                    step=10,
                    info="建议50-100，过大可能超出token限制"
                )
                concurrency = gr.Slider(
                    label="并行批次数",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    info="同时处理的批次数，越大越快但API压力越大"
                )

            with gr.Row():
                submit_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                test_btn = gr.Button(f"🧪 快速测试（前{TEST_COUNT}条）", variant="secondary", size="lg")
                cancel_btn = gr.Button("⏹️ 取消", variant="stop", size="lg")

        # 右侧：结果和说明
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("📈 分析结果"):
                    output_file = gr.File(label="下载结果文件")
                    output_report = gr.Markdown(label="分析报告")

                with gr.TabItem("📊 评分标准"):
                    gr.Markdown("""
                    | 标签 | 分数 | 说明 |
                    |------|------|------|
                    | 泛体育运动 | 0.9 | 足球、篮球、健身、户外等明确运动内容 |
                    | 年轻潮流 | 0.7 | 潮牌、数码、游戏、潮玩 |
                    | 社交场景 | 0.7 | 聚会、啤酒、夜宵社交 |
                    | 夜间经济 | 0.6 | 宵夜、深夜、晚间活动 |
                    | 活动参与型 | 0.5 | 薅羊毛、抢优惠、高活跃 |
                    | 家庭亲子 | 0.3 | 亲子、儿童餐 |
                    | 通勤日常 | 0.1 | 早餐、午餐、日常刚需 |
                    | 其他 | 0.2 | 未匹配到以上标签 |

                    **阈值：≥ 0.4 为匹配（1）**

                    **说明：**
                    - 核心标签：泛体育运动、年轻潮流、社交场景、夜间经济、活动参与型
                    - 低相关标签：家庭亲子、通勤日常
                    - 其他：未匹配标签，0.2分，低于阈值
                    """)

                with gr.TabItem("ℹ️ 使用说明"):
                    gr.Markdown("""
                    **操作步骤：**
                    1. 上传Excel文件
                    2. 选择Sheet和文案标题所在列（从1开始计数）
                    3. 填写API配置
                    4. 点击"快速测试"验证效果，确认OK后点"开始分析"
                    5. 等待处理完成，下载结果

                    **列号说明：**
                    - 从1开始计数，第1列=1，第2列=2，以此类推
                    - 例如：消息ID在第1列，消息类型在第2列，创建时间在第3列，文案标题在第4列，则填4

                    **输出说明：**
                    - 人群匹配：0或1（≥0.4为1）
                    - 匹配度分数：0.0-1.0
                    - 匹配理由：基于文案关键词的简短说明

                    **注意事项：**
                    - 每批处理条数建议50-100
                    - 并行批次数建议3-5，API压力大时调低
                    - 并行处理可大幅缩短时间（3并发≈3倍速）
                    - 如遇限流会自动重试
                    - 可随时点击"取消"按钮停止
                    """)

    # 事件处理
    def update_sheet_dropdown(file):
        if file is None:
            return gr.Dropdown(choices=[])
        sheets = get_sheet_names(file)
        return gr.Dropdown(choices=sheets, value=sheets[0] if sheets else None)

    refresh_btn.click(fn=update_sheet_dropdown, inputs=[file_input], outputs=[sheet_dropdown])
    file_input.change(fn=update_sheet_dropdown, inputs=[file_input], outputs=[sheet_dropdown])

    submit_btn.click(
        fn=process_excel,
        inputs=[file_input, sheet_dropdown, title_col_idx, api_key, base_url, model, batch_size, concurrency],
        outputs=[output_file, output_report]
    )

    test_btn.click(
        fn=lambda *args: process_excel(*args, test_mode=True),
        inputs=[file_input, sheet_dropdown, title_col_idx, api_key, base_url, model, batch_size, concurrency],
        outputs=[output_file, output_report]
    )

    cancel_btn.click(fn=cancel_processing, outputs=[output_report])

# 启动
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
