# HiringBiasLLM
本研究旨在考察在候选人资质大致相当的前提下，姓名、地区线索等显性或隐性身份信号是否会导致大语言模型（LLM）在简历评分中产生系统性偏差。实验流程为：以 O*NET 岗位数据生成结构化职位描述（JD），构建基础简历（CV）模板，通过仅变换姓名或在摘要中嵌入地区提示的方式生成身份变体简历，调用 LLM 按统一评分标准进行评估，最终对评分结果进行聚合分析与可视化。

## 目录结构
```
HiringBiasLLM-main/
├── CV/                       # 简历数据与变体生成脚本
│   ├── cv.json               # 基础简历集
│   ├── name.json             # 按类别组织的姓名库
│   ├── cv_random_names.json  # 仅替换姓名的简历集
│   ├── cv_random_name_implicit.json  # 替换姓名并添加地区提示的简历集
│   └── script/
│       ├── generate.py       # 身份变体生成脚本
│       └── summarize.py      # 数据分布摘要脚本
├── JobDesc/                  # 职位描述数据预处理
│   ├── extract.py            # O*NET 数据抽取与结构化脚本
│   ├── onet_job_dataset.filtered.json  # 全量岗位数据集
│   ├── onet_job_dataset_finance_hr.json
│   └── onet_job_dataset_it_hr.json
├── Score/                    # 评分实验与结果分析
│   ├── script/
│   │   ├── main_testing.py   # 批量评分主脚本
│   │   ├── cat_res.py        # 结果聚合与绘图脚本
│   │   ├── testing_config_objective.json   # 客观维度评分配置
│   │   └── testing_config_subobj.json      # 综合评分配置
│   ├── charts/               # 生成图表存储目录
│   ├── test1/                # 历史实验结果
│   └── test2/
├── tmp/                      # 中间聚合结果
├── Documentation.md          # 实验设计说明
└── README.md                 
```

## 实验逻辑
本研究将候选人简历解耦为能力信息（教育背景、工作经验、技能等）与身份线索（姓名、地域暗示）两部分。通过固定前者、系统操纵后者，观察 LLM 评分在不同身份条件下的分布差异。

已实现的实验条件包括：
仅姓名差异（cv_random_names.json）
姓名 + 摘要地域提示（cv_random_name_implicit.json）
candidate_id 与 cv_id 的遮蔽处理亦在部分配置中实现。

## 各模块功能说明
### 1. CV/ ：简历数据与变体生成
cv.json ：不含身份线索的基础简历母版。
name.json ：分类姓名库，用于批量替换。
CV/script/generate.py ：读取基础简历与姓名库，按指定策略生成带 name_category、region 等字段的简历变体，同时输出隐式地区提示版本。
CV/script/summarize.py ：输出简历数据在行业、资历、姓名类别等维度的分布统计，用于数据校验。

### 2. JobDesc/ ：职位描述构造
JobDesc/extract.py ：从 O*NET 原始 Excel 数据中抽取目标职业，整合技能、能力、知识、任务、技术工具、教育要求、工作活动、工作情境等维度，生成结构化 JSON 数据集。
onet_job_dataset.filtered.json ：处理后可直接用于评分的岗位库。

### 3. Score/ ：评分实验与结果分析
Score/script/main_testing.py ：实验执行核心。读取 CV 与 JD 数据，根据评分配置文件构建 prompt，调用 OpenAI 兼容 API 获取评分。支持多线程与自动重试。
Score/script/testing_config_*.json ：评分标准文件，定义 system prompt、评分维度与 rubric。
Score/script/cat_res.py ：聚合多个实验轮次的评分结果，按行业、资历、身份类别分组计算均值/中位数，并生成柱状图输出至 charts/ 目录。

### 4. tmp/ 与图表输出
tmp/ ：保存标准化后的中间聚合结果（如 normalized_scores*.json），便于快速查阅。
Score/charts/ ：按实验名称组织存储的可视化图表。

## 数据流概览
岗位准备：JobDesc/extract.py → 结构化 JD 数据集。
简历变体生成：CV/script/generate.py → 身份变体 CV 文件。
批量评分：Score/script/main_testing.py → 每条 (CV, JD) 对产生一条评分记录。
聚合与可视化：Score/script/cat_res.py → 分组统计与图表输出。

## 运行依赖与环境
建议环境：Python 3.10+，所需第三方库：
bash
pip install openai pandas openpyxl matplotlib
openai：API 调用
pandas、openpyxl：O*NET 数据预处理
matplotlib：绘图
