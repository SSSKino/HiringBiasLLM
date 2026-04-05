# Research Documentation
Investigating Regional Hiring Bias in Large Language Models

## 1. Research Question

When LLMs are employed as resume screening tools, do they produce systematically different evaluation scores for candidates from different regional or ethnic backgrounds, given otherwise equivalent qualifications?

## 2. Industry Selection

To improve diversity and representativeness, four distinct industries were selected for investigation. These industries were chosen to represent a range of professional domains with different skill requirements, qualification standards, and cultural expectations:

- Information Technology (IT)
- Finance
- Human Resource (HR)
- Law

## 3. Job Descriptions Collection

For each of the four target industries, approximately 15–20 authentic job descriptions were sourced from O\*NET OnLine (https://www.onetonline.org/), an occupational information database maintained by the U.S. Department of Labour. All collected job descriptions were structured and stored in JSON format to facilitate programmatic processing.

## 4. Resume (CV) Generation

Synthetic resumes were generated for each industry across three professional seniority levels to capture potential bias variation at different career stages:

| Seniority Level | Description | CVs per Industry |
|-----------------|-------------|------------------|
| Junior | Entry-level, 0–3 years of experience | 3 |
| Senior | Mid-career, 5–10 years of experience | 3 |
| Managerial | Leadership-level, 10+ years of experience | 3 |

This yields a total of 9 unique resumes per industry (3 seniority levels × 3 variants), and 36 base resumes across all four industries.

## 5. Regional Identity Groups

Six geographic and ethnic regions were selected to represent a broad cross-section of global diversity:

- **American** — Names commonly associated with the United States
- **European** — Names commonly associated with Western/Central Europe
- **Indian** — Names commonly associated with the Indian subcontinent
- **Chinese** — Names commonly associated with China and Chinese diaspora
- **Middle Eastern** — Names commonly associated with the Middle East
- **African** — Names commonly associated with Africa

Each base resume is replicated across all six regional groups, yielding 6 variants per resume and a total of 216 resume instances (36 base resumes × 6 regions) before accounting for disclosure levels.

## 6. Regional Disclosure Levels

For this research, we designed a three-tiered disclosure framework to examine how the explicitness of regional identity cues affects LLM scoring behaviour.

| Level | Mechanism | Description |
|-------|-----------|-------------|
| Level 1: Implicit | Name Only | Regional identity is implied solely through the candidate's name. |
| Level 2: Summary Statement | Summary Statement | In addition to the name, the resume's professional summary includes a self-disclosure phrase (e.g., "Originally from [region]..."). |
| Level 3: Explicit | Dedicated Field | A "Region" field is explicitly added to the resume, making the candidate's background a structured data point. |

With three disclosure levels applied to 216 resume instances, the full experimental matrix comprises 648 individual evaluation instances (36 base CVs × 6 regions × 3 disclosure levels).

## 7. LLM Model Selection

To assess whether bias patterns are model-specific or systemic across architectures, the following LLMs have been provisionally selected for evaluation:

- Qwen
- GPT (OpenAI)
- Gemini
- Claude

## 8. Prompt and Scoring Rubric Design (In Progress)

## 9. Weekly Progress

- Defined the primary research question and scope
- Selected four target industries: IT, Finance, HR, Law
- Designed the three-tier regional disclosure framework: name only, summary statement and explicit field
- Identified six regional identity groups for cross-cultural comparison
- Collected 15–20 JDs per industry from O\*NET Online and standardised all JDs into a structured JSON format
- Generated 9 synthetic CVs per industry (3 seniority levels × 3 variants), totalling 36 base CVs
- Applied regional identity names across all base CVs for the six target regions
- Conducted an evaluation using the Qwen API to preliminarily validate the prompt design and scoring rubric; initial results have been collected and are currently under analysis
