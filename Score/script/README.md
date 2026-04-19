# CV评分系统 (CV Scoring System)

## 项目概述

这是一个基于大型语言模型(LLM)的**CV评分和招聘偏见研究**系统。用于自动评分候选人简历与职位要求的匹配度，支持多维度评分和批量处理。

## 核心功能

- **自动CV评分**: 使用LLM对候选人简历进行6维度评分
- **多实验对比**: 支持3个独立实验对比不同隐藏策略的影响
- **批量处理**: 支持多线程并行处理CV-职业对
- **数据隐私**: 支持隐藏敏感信息(候选人ID、CV ID)
- **API模式和审核模式**: 支持实际API调用或仅生成请求负载用于审核
- **重试机制**: 自动检测和重试格式错误的响应

## 模块架构

### `CVScoringUtils` 类

核心工具类，包含9个静态方法：

| 方法                                                | 功能                             |
| ------------------------------------------------- | ------------------------------ |
| `load_json(path)`                                 | 加载JSON文件（CV、职业数据集、配置）          |
| `get_JD_INFO(occupations, limit)`                 | 提取职位信息，压缩职业数据到关键字段             |
| `decompose_JD_to_Batch(items, batch_size)`        | 将职业列表分块处理                      |
| `load_prompt(path)`                               | 加载提示词配置文件（包含系统消息、指令、输出格式、评分标准） |
| `build_prompt(candidates, standard, cfg)`         | 构建发送给LLM的完整提示词                 |
| `score(client, model, candidates, standard, cfg)` | 调用LLM API进行CV评分                |
| `check_APIReturn_Format(result)`                  | 验证LLM返回的评分格式是否符合6维度标准          |
| `mask_CV_KEY(cv, mask_candidate_id, mask_cv_id)`  | 隐藏CV中的敏感字段                     |
| `attach_ids(result, candidate_id, cv_id)`         | 在结果中重新附加ID                     |

### 关键函数

#### `run_experiment(...)`

执行单个评分实验的完整流程：

- 加载CV和职业数据
- 多线程处理每个CV-职业对
- 隐藏敏感信息
- 构建提示词并调用LLM
- 验证返回格式（失败则重试）
- 收集和返回结果

#### `main()`

程序入口，处理：

- 命令行参数解析
- 数据加载和验证
- 3个实验的定义和执行
- 多轮运行管理
- 结果输出到JSON文件

## 三个实验设置

| 实验   | CV类型 | 隐藏策略          | 目的          |
| ---- | ---- | ------------- | ----------- |
| exp1 | 非隐式  | 隐藏候选人ID和CV ID | 基准实验        |
| exp2 | 隐式   | 隐藏候选人ID和CV ID | 对比隐式CV的效果   |
| exp3 | 隐式   | 仅隐藏CV ID      | 保留候选人ID进行对比 |

## 安装依赖

```bash
pip install openai
```

## 使用方法

### 基础用法

```bash
python main_testing.py --experiment 1 \
  --industry IT \
  --JD_NUM 3 \
  --CV_NUM 5 \
  --model qwen-max \
  --api-key sk-xxxx \
  --output results.json
```

### 参数说明

| 参数                  | 类型      | 默认值                                 | 说明                             |
| ------------------- | ------- | ----------------------------------- | ------------------------------ |
| `--cv`              | str     | `.\CV\cv_random_names.json`         | 非隐式CV文件路径                      |
| `--cv-implicit`     | str     | `.\CV\cv_random_name_implicit.json` | 隐式CV文件路径                       |
| `--jobs`            | str     | `.\onet_job_dataset.filtered.json`  | O*NET职业数据集路径                   |
| `--prompt`          | str     | `.\testing_config.json`             | 提示词配置文件路径                      |
| `--output`          | str     | `None`                              | 输出文件路径（单实验时使用）                 |
| `--output-prefix`   | str     | `cv_scores`                         | 输出文件名前缀（多实验时使用）                |
| `--model`           | str     | 环境变量LLM_MODEL                       | 使用的LLM模型名称                     |
| `--api-key`         | str     | 环境变量LLM_API_KEY                     | LLM API密钥                      |
| `--base-url`        | str     | `https://llm.eulerai.au`            | LLM API地址                      |
| `--experiment`      | str     | **必需**                              | 运行哪个实验: `1`, `2`, `3`, 或 `all` |
| `--industry`        | str     | `IT`                                | 评分的行业类别                        |
| `--JD_NUM`          | int/str | `3`                                 | 职位数量（`all`表示全部）                |
| `--CV_NUM`          | int     | `None`                              | CV数量限制（None表示全部）               |
| `--jobs-batch-size` | int     | `None`                              | 职业分批大小                         |
| `--review-output`   | str     | `None`                              | 审核模式输出路径（设置则不调用API）            |
| `--runs`            | int     | `1`                                 | 独立运行次数                         |
| `--thread`          | int     | `1`                                 | 并行工作线程数                        |

## 评分维度（6维度）

LLM对每个CV-职位对进行以下6个维度的评分（0-100分）：

1. **skill_match** - 技能匹配度
2. **experience_match** - 经验匹配度
3. **education_match** - 教育背景匹配度
4. **communication_and_collaboration** - 沟通合作能力
5. **execution_compliance_reliability** - 执行力和可靠性
6. **role_context_adaptability** - 角色适应性

每个维度需要包含：

- `score`: 数值评分（0-100）
- `reasoning`: 评分理由

## 输出格式

### 结果文件结构

```json
{
  "industry": "IT",
  "experiments": [
    {
      "name": "exp1_mask_candidate_and_cv_nonimplicit",
      "cv_path": "path/to/cv.json",
      "mask_candidate_id": true,
      "mask_cv_id": true,
      "results": [
        {
          "scores": {
            "skill_match": {"score": 85, "reasoning": "..."},
            "experience_match": {"score": 90, "reasoning": "..."},
            ...
          },
          "candidate_id": "C001",
          "cv_id": "CV001",
          "industry": "IT"
        }
      ]
    }
  ]
}
```

### 审核模式输出

设置 `--review-output` 时，输出仅包含请求负载，不调用API：

```json
{
  "industry": "IT",
  "experiments": [
    {
      "name": "exp1_...",
      "cv_path": "path/to/cv.json",
      "payloads": [
        {
          "candidates": [...],
          "job_standard": [...],
          "industry": "IT"
        }
      ]
    }
  ]
}
```

## 使用示例

### 示例1: 单实验，5秒间隔多线程评分

```bash
python main_testing.py \
  --experiment 1 \
  --industry IT \
  --JD_NUM 3 \
  --CV_NUM 10 \
  --model gpt-4 \
  --api-key sk-xxxx \
  --thread 2 \
  --output results_exp1.json
```

### 示例2: 运行所有实验，每个实验独立client

```bash
python main_testing.py \
  --experiment all \
  --industry HR \
  --JD_NUM all \
  --CV_NUM 50 \
  --model qwen-max \
  --api-key sk-xxxx \
  --output-prefix hiring_bias_study \
  --runs 2
```

### 示例3: 审核模式，仅生成请求负载

```bash
python main_testing.py \
  --experiment 1 \
  --industry IT \
  --JD_NUM 5 \
  --CV_NUM 20 \
  --review-output payloads_review.json
```

## 配置文件格式 (testing_config.json)

```json
{
  "system_msg": "You are an expert HR recruiter...",
  "eval_instructions": "Please evaluate the following CVs against the job standard:\n\n[JD_CONTENT]\n\nCandidates:\n[CV_CONTENT]\n\nReturn results in this format:\n[OUTPUT_FORMAT]",
  "output_format": {
    "skill_match": {
      "score": "number 0-100",
      "reasoning": "string"
    }
  },
  "rubric": {
    "skill_match": "评分标准详情...",
    ...
  }
}
```

## 环境变量

可通过以下环境变量设置默认值，避免每次都输入参数：

```bash
# 设置LLM配置
export LLM_MODEL=qwen-max
export LLM_API_KEY=sk-xxxx
export LLM_BASE_URL=https://llm.eulerai.au
```

## 工作流程

```
命令行参数解析
    ↓
加载CV文件、职业数据集、提示词配置
    ↓
确认实验配置（用户输入 y/n）
    ↓
[FOR 每个run]
  └─ [FOR 每个实验]
     └─ 创建新client
     └─ 构建CV-职业对任务列表
     └─ [多线程] 处理每个任务
        ├─ 隐藏敏感信息
        ├─ 构建提示词
        ├─ 调用LLM评分
        ├─ 验证返回格式
        └─ [格式错误则重试]
     └─ 输出结果到JSON
```

## 线程间隔

多线程模式下，每个任务启动间隔为**5秒**，用于：

- 避免API速率限制
- 分散API调用负载
- 提高成功率

## 错误处理

系统包含以下错误处理：

1. **文件不存在**: 退出并提示文件路径
2. **JSON解析错误**: 显示错误行号和原因
3. **格式检查失败**: 自动重试一次
4. **API调用中断**: 保存已有结果后退出（Ctrl+C）
5. **缺少必需参数**: 退出并列出缺少的参数

## 常见问题

**Q: 如何加快评分速度？**  
A: 增加 `--thread` 参数的值，例如 `--thread 5`

**Q: 如何测试提示词效果而不消耗API配额？**  
A: 使用 `--review-output payloads.json` 生成请求负载进行审核

**Q: 如何重新运行失败的部分？**  
A: 使用 `--CV_NUM` 限制数量或 `--JD_NUM` 限制职位数，然后调整 `--output-prefix`

**Q: 隐式CV和非隐式CV有什么区别？**  
A: 隐式CV去除了明显的个人身份信息，用于对比是否会减少招聘偏见

## 许可证

[待定]

## 联系方式

[待定]
