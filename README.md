# HiringBiasLLM

本研究旨在考察在候选人资质大致相当的前提下，姓名、地区线索等显性或隐性身份信号是否会导致大语言模型（LLM）在简历评分中产生系统性偏差。

## 目录结构

```
githubbk/
├── script/                           # 所有脚本程序
│   ├── generate.py                   # CV变体生成脚本：根据基础简历和姓名库生成带身份变体的简历
│   ├── extract.py                    # O*NET数据抽取脚本：从原始数据中提取职位描述
│   ├── main_testing.py               # 批量评分主脚本：调用LLM进行简历-职位评分
│   ├── analyze.py                    # 分析与可视化脚本：评分结果聚合、标准化、分组统计
│   └── summarize.py                  # 数据分布摘要脚本：输出简历数据的统计信息
│
├── json/                             # 所有数据文件
│   ├── cv.json                       # 基础简历集（不含身份线索）
│   ├── cv_random_names.json          # 仅替换姓名的简历变体
│   ├── cv_random_name_implicit.json  # 替换姓名+地区提示的简历变体
│   ├── name.json                     # 按类别组织的姓名库
│   ├── onet_job_dataset.filtered.json        # 全量职位数据集（处理后）
│   ├── onet_job_dataset_it_hr.json           # IT行业职位数据
│   ├── onet_job_dataset_finance_hr.json      # 金融行业职位数据
│   └── prompt.json                   # LLM评分提示词配置
│
├── Documentation.md                  # 详细的实验设计与逻辑说明
├── README.md                         # 本文件
└── 反馈总结.docx                    # 研究反馈记录
```


## 核心脚本功能说明

### 1. generate.py - CV变体生成
**功能**：读取基础简历和姓名库，生成带身份变体的简历集合

**输入**：
- `--cv` (default: `json/cv.json`) - 基础简历集
- `--names` (default: `json/name.json`) - 姓名库（支持按类别组织）
- `--n` (default: 1) - 每条简历生成的变体数量

**输出**：
- `json/cv_random_names.json` 或指定的输出路径

**关键字段**：
- `name_category`: 姓名类别（地区、民族等分类）
- `region`: 对应的地区标签
- `candidate_id`: 格式为 `{industry}_{seniority}_{region}`
- `cv_id`: 原始简历ID

**使用示例**：
```bash
python script/generate.py --cv json/cv.json --names json/name.json --output json/cv_random_names.json --seed 42
```

### 2. extract.py - O*NET数据抽取
**功能**：从O*NET原始数据中抽取职位信息，生成结构化职位描述

**输入**：O*NET Excel数据源

**输出**：
- `json/onet_job_dataset.filtered.json` - 全量职位数据
- `json/onet_job_dataset_it_hr.json` - IT行业职位
- `json/onet_job_dataset_finance_hr.json` - 金融行业职位

### 3. main_testing.py - 批量评分
**功能**：调用LLM API对简历进行评分，支持多轮实验和自动重试

**输入**：
- `--cv` (default: `json/cv_random_names.json`) - 简历集
- `--jobs` (default: `json/onet_job_dataset.filtered.json`) - 职位集
- `--prompt` (default: `testing_config.json`) - 评分标准配置
- `--model` - LLM模型名称（从环境变量获取）
- `--api-key` - API密钥（从环境变量获取）
- `--industry` (default: `IT`) - 筛选行业
- `--JD_NUM` (default: 3 or all)  
  使用的职位数量，或 "all" 表示全部
- `--JD_START` (default: 1)  
  JD 起始索引（1-based）
- `--experiment` (required: 1 / 2 / 3 / all)  
  选择运行的实验类型：

  - 1：仅姓名变化（non-implicit CV，隐藏 candidate_id / cv_id）
  - 2：姓名 + 隐式身份信息（implicit CV，隐藏 candidate_id / cv_id）
  - 3：隐式 CV + 显式 region（隐藏 cv_id）
  - all：依次运行 exp1 → exp2 → exp3

**输出**：
- `cv_scores_exp1_run1.json`, 如果多次运行记得将结果保存到别处防止覆盖。

**评分维度**（6个维度）：
- skill_match: 技能匹配度
- experience_match: 经验匹配度
- education_match: 教育匹配度
- communication_and_collaboration: 沟通协作能力
- execution_compliance_reliability: 执行力和可靠性
- role_context_adaptability: 角色适应性

**使用示例**：
```bash
python script/main_testing.py \
  --cv json/cv_random_names.json \
  --jd json/onet_job_dataset_it_hr.json \
  --prompt-config testing_config.json \
  --model gpt-4 \
  --industry IT
```

### 4. analyze.py - 评分分析与可视化
**功能**：规范化评分结果、按地区分组统计、生成可视化图表

**处理流程**：
1. 加载所有 `cv_scores_*.json` 评分文件
2. 规范化评分（每个维度除以6，总分为平均值）
3. 按候选人ID提取地区信息
4. 生成规范化的JSON文件（`normalized/` 目录）
5. 生成两类图表：
   - **Chart 1**: 按地区显示平均评分
   - **Chart 2**: 按维度显示评分变异性（max-min差值及来源文件）

**输入**：
- `--input` (default: `.`) - 分数文件或目录
- `--score-glob` (default: `*score*.json`) - 文件匹配模式

**输出**：
- `normalized/`: 规范化的评分文件
- `stability_charts/`: 生成的图表
  - `regions_by_source_*.png` - 各数据源的地区评分图
  - `score_variability_by_segment.png` - 评分变异性分析图

**使用示例**：
```bash
python script/analyze.py --input . --chart-dir charts --normalized-dir normalized
```

### 5. summarize.py - 数据分布统计
**功能**：输出简历数据在行业、资历、姓名类别等维度的分布统计

**用途**：验证生成的简历数据分布是否符合预期

---

## 实验工作流

```
1. 数据准备
   ├─ 获取基础简历 (cv.json) 和姓名库 (name.json)
   └─ 获取O*NET职位数据

2. 简历变体生成
   └─ python script/generate.py
      生成 cv_random_names.json 和其他变体

3. 职位描述准备
   └─ python script/extract.py
      生成 onet_job_dataset.filtered.json

4. 批量评分
   └─ python script/main_testing.py
      生成 cv_scores_exp1_run1.json 等评分文件
      （支持多轮运行，自动生成新的run文件）

5. 结果分析
   └─ python script/analyze.py
      输出规范化数据和图表
      - normalized_*.json 文件
      - 地区评分对比图
      - 维度变异性分析图
```

## 环境要求

Python 3.10+

**依赖包**：
```bash
pip install openai pandas openpyxl matplotlib
```

- `openai`: LLM API调用
- `pandas`: 数据处理
- `openpyxl`: Excel数据读取
- `matplotlib`: 图表生成

**环境变量**：
```bash
export LLM_MODEL="gpt-4"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
```

## 关键概念

- **candidate_id**: 候选人标识，格式 `{industry}_{seniority}_{region}`，用于提取地区信息进行分组分析
- **规范化评分**: 每个维度分数 ÷ 6（单个维度范围0-16.67），总分为六个维度的平均值（范围0-100）
- **地区变异性**: 同一地区内不同评分文件产生的最大-最小差值，用于检测评分偏差
- **评分维度**: 6个评分维度代表不同能力方面，总分为其平均值
