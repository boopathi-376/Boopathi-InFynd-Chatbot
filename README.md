OpenBot Validation & Semantic Retrieval System

OpenBot is a FastAPI + Qdrant based intelligent query understanding and validation system.
It uses SentenceTransformer (intfloat/e5-base-v2) as its local embedding model — (qwen2.5:7b)LLMs for validation.

⚙️ Tech Stack Overview
Component	Purpose	Notes
FastAPI	REST API layer for validation and reasoning	Async-capable
SentenceTransformer (E5-Base-V2)	Local text embedding model	Used for both query understanding and matching
Qdrant	Vector DB for fast semantic retrieval	Runs in Docker
Torch + NumPy	Vector math and device acceleration	Uses GPU if available
📁 Folder Structure
OPENBOT/
├── scripts/
│   ├── api_server.py             # Main FastAPI-based validation API
│   ├── embed_to_qdrant.py        # JSON → embeddings → Qdrant ingestion
│
├── data/                         # All source JSON data for Qdrant
│   ├── company_type.json
│   ├── job_function.json
│   └── ...
│
├── local_models/
│   └── e5-base-v2/               # Local embedding model (downloaded once)
│
├── requirements.txt
└── README.md



Python 3.10+

Docker (for Qdrant)

4GB+ RAM (recommended: 8GB)

Optional GPU (CUDA for speed)

2️⃣ Install Dependencies
pip install -r requirements.txt

requirements.txt
fastapi==0.115.0
uvicorn==0.30.0
qdrant-client==1.10.1
sentence-transformers==3.0.1
torch>=2.2.0
numpy>=1.26.0
pydantic>=2.8.0
cachetools
regex
json5

3️⃣ Run Qdrant Database via Docker
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage:z \
  qdrant/qdrant


Check dashboard:
👉 http://localhost:6333/dashboard

4️⃣ Embed Data into Qdrant

Put your JSON data files in /data, then run:

python scripts/embed_to_qdrant.py


This script:

Loads the E5-Base-V2 model

Embeds all JSON data fields

Uploads to Qdrant (batch mode)

5️⃣ Start the API Server
python scripts/api_server.py


The FastAPI server runs at:

http://localhost:8000


Interactive docs available at:

http://localhost:8000/docs

🧾 Example Query Flow
Input
{
  "query": "List all Fintech companies in the UK"
}

Processing Steps

The query is embedded using E5-Base-V2

Qdrant returns the top-matching filters

The server builds a semantic response + suggestions

{
  "query": "Top 10 companies in UK",
  "qdrant_result": {
    "company_type": [
      "UK Establishment Company",
      "United Kingdom Societas",
      "Overseas Company",
      "United Kingdom Economic Interest Grouping",
      "Investment Company with Variable Capital"
    ],
    "sic_code_description": [
      "99999 | Dormant company",
      "20110 | Manufacture of industrial gases",
      "20301 | Manufacture of paints, varnishes and similar coatings, mastics and sealants",
      "64209 | Activities of other holding companies n.e.c.",
      "69102 | Solicitors"
    ],
    "company_status": [
      "Private Limited",
      "Liquidation",
      "Registered",
      "Other",
      "Active"
    ],
    "marketable_flag_p": [
      "Emailable | people_email_flag",
      "Phonable | field_5"
    ],
    "job_description": [
      "UK Business Manager",
      "UK Managing Director",
      "Company Directory",
      "UK Chief Technology Officer",
      "Managing Director UK"
    ],
    "job_function": [
      "HR",
      "Uncategorised",
      "Marketing",
      "RND",
      "Compliance"
    ],
    "includes_p": [
      "Company Email | company_email_flag",
      "Company Phone | phone_flag",
      "CRN | registration_number_1",
      "Mailable | mailable_flag"
    ],
    "turnover_range": [
      "100K to 250K",
      "Upto 100K",
      "10M to 25M",
      "5M to 10M",
      "250K to 500K"
    ],
    "cd_geographyCountries": [
      "United Kingdom | united_kingdom",
      "New Zealand | new_zealand",
      "Australia | australia",
      "Canada | canada",
      "Sweden | sweden"
    ],
    "country": [
      "United Kingdom | united_kingdom",
      "New Zealand | new_zealand",
      "Australia | australia",
      "Canada | canada",
      "Sweden | sweden"
    ],
    "hiring_ind": [
      "Yes | True",
      "No | False"
    ],
    "marketable_flag_c": [
      "Emailable | company_email_flag",
      "Mailable | mailable_flag",
      "Phonable | phone_flag"
    ],
    "post_code": [
      "BS10 7BN",
      "SW10 9BN",
      "BS10 7TN",
      "BT10 0BN",
      "SW10 0BN"
    ],
    "Subindustry": [
      "NHS | Health Care Providers & Services | Healthcare",
      "Wholesale British Food | Food & Beverage Industry | Consumer Goods & Services",
      "Pub Company | Pubs, Bars & Inns | Hospitality",
      "Oil And Gas Companies | Oil, Gas & Consumable Fuels | Utilities & Energy",
      "Record Companies | Music Industry | Media, Publishing & Entertainment"
    ],
    "account_category": [
      "Audit Exemption Subsidiary",
      "Filing Exemption Subsidiary",
      "Total Exemption Small",
      "Micro Entity",
      "Total Exemption Full"
    ],
    "location_type": [
      "Head Office",
      "Single Site",
      "Branch"
    ],
    "supressionType": [
      "Inclusion | inclusion",
      "Exclusion | exclusion"
    ],
    "cd_companyMaximumAge": [
      "5-10_year | 5-10 year",
      "100_years | 100 years+",
      "11-20_year | 11-20 year",
      "0-1_year | 0-1 year",
      "51-100_year | 51-100 year"
    ],
    "technologies": [
      "pycharm | PyCharm",
      "topdesk | TOPdesk",
      "cadvisor | CAdvisor",
      "bigcommerce | BigCommerce",
      "data_types_lists_tuples_dictionaries_sets | Data Types (Lists, Tuples, Dictionaries, Sets)"
    ],
    "town_county_country": [
      "Deal | Kent | England",
      "Cambridge | Cambridgeshire | England",
      "Leyland | Lancashire | England",
      "London | London | England",
      "Manchester | Lancashire | England"
    ],
    "job_title_level": [
      "Company Secretary",
      "Managing Director/CEO",
      "Founder/Owner",
      "VP level",
      "C Level"
    ],
    "cd_sicCode": [
      "Oil And Gas Companies | Oil And Gas Companies",
      "Property & Estate Management Companies | Property & Estate Management Companies",
      "Plant & Machinery Manufacturers | Plant & Machinery Manufacturers",
      "Theatre Companies | Theatre Companies",
      "Financial Planning & Investment Management Companies | Financial Planning & Investment Management Companies"
    ],
    "includes_c": [
      "Company Linkedin | linkedin_1",
      "CRN | registration_number_1"
    ],
    "employee_range": [
      "Uncategorised",
      "6 to 10",
      "500 to 999",
      "1000 to 5000",
      "100 to 199"
    ]
  },
  "llm_validated_output": {
    "intent": "Top 10 companies in UK",
    "validated_filters": {
      "cd_geographyCountries": [
        "United Kingdom | united_kingdom"
      ]
    },
    "reasoning": "The query asks for top companies in the UK, and 'cd_geographyCountries' contains 'United Kingdom | united_kingdom', which matches the intent."
  },
  "suggestions": {
    "company_type": [
      "UK Establishment Company",
      "United Kingdom Societas"
    ],
    "job_description": [
      "UK Business Manager",
      "UK Managing Director",
      "Company Directory"
    ],
    "cd_geographyCountries": [
      "United Kingdom | united_kingdom"
    ],
    "country": [
      "United Kingdom | united_kingdom"
    ],
    "Subindustry": [
      "NHS | Health Care Providers & Services | Healthcare",
      "Wholesale British Food | Food & Beverage Industry | Consumer Goods & Services",
      "Pub Company | Pubs, Bars & Inns | Hospitality"
    ],
    "cd_sicCode": [
      "Oil And Gas Companies | Oil And Gas Companies",
      "Property & Estate Management Companies | Property & Estate Management Companies"
    ]
  },
  "processing_time_seconds": 16.93,
  "mode": "live"
}
