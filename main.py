from src.parser import extract_text
from src.extract_skills import match_skills
from src.translator import detect_language, translate_to_english
import logging 
import time
from os import listdir, path
from os.path import isfile
from typing import Set, Tuple, List


def translation_pipeline(skills) -> Set[str]:
    translated_skills = set()
    count = 0
    i = 0
    to_translate = []
    lenght = 0
    while count < len(skills):
        detection = detect_language(skills[i])
        if detection:
            to_translate += [skills[i]]
            lenght += len(skills[i])
        elif not detection:
            translated_skills.add(skills[i])
        if lenght >= 100 or (to_translate and count == len(skills)-1):
            lenght = 0
            translated_chunk, _ = translate_to_english(" ; ".join(i for i in to_translate))
            for skill in translated_chunk.split(" ; "):
                translated_skills.add(skill.strip())
            to_translate = []
        i += 1
        count += 1
        print(translated_skills)
    return translated_skills


def main():
    start_time = time.time()
    TRANSLATED_SKILL_DB = list(translation_pipeline(skills=SKILLS_DB))
    print(TRANSLATED_SKILL_DB)
    PATH = r"C:/Users/sanya/Desktop/pet projects/job matcher/resume_extractor/Resumes"
    results = []
    for file in listdir(PATH):
        if isfile(path.join(PATH, file)):
            with open(f"{PATH}/{file}", "rb") as f:
                text = extract_text(f.read(), file)
                need_to_translate = detect_language(text)
                translated_resume = None
                if need_to_translate:
                    logging.info("🌐 Detected non-English resume. Translating to English for skill extraction...")
                    translated_resume, was_translated = translate_to_english(text, detect_language(text))
                results = match_skills(translated_resume, TRANSLATED_SKILL_DB) if translated_resume else match_skills(text, TRANSLATED_SKILL_DB)
                with open(f"{PATH}/results.txt", "a", encoding="utf-8") as res_file:
                    res_file.write(f"Resume: {file}\n")
                    res_file.write("Top skill matches:\n")
                    for skill, score in results:
                        res_file.write(f"  {skill}: {score:.3f}\n")
                logging.info(f"Processed '{file}' with {len(results)} matched skills.")
    logging.info(f"All resumes processed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("Starting resume processing pipeline...")

    # Sample skills database
    SKILLS_DB = [
        "Python", "JavaScript", "JS", "TypeScript", "TS", "Java", "C++", "C#", "C", "Go", "Golang", "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R", "SQL", "PL/SQL", "Bash", "Shell", "PowerShell", "MATLAB", "Julia", "Dart", "Objective-C", "Perl", "Haskell", "Lua", "Assembly", "x86", "ARM",
        "React", "Vue.js", "Vue", "Angular", "Svelte", "Next.js", "Nuxt.js", "HTML5", "CSS3", "Tailwind CSS", "Bootstrap", "Sass", "SCSS", "LESS", "Webpack", "Vite", "Babel", "npm", "yarn", "pnpm", "jQuery", "Three.js", "D3.js", "WebGL", "WebAssembly", "Wasm",
        "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot", ".NET Core", "ASP.NET", "Ruby on Rails", "Laravel", "Phoenix", "Elixir", "GraphQL", "REST", "gRPC", "SOAP", "OpenAPI", "Swagger", "Postman", "Apollo", "tRPC",
        "PostgreSQL", "Postgres", "MySQL", "MongoDB", "Redis", "SQLite", "Oracle", "SQL Server", "Cassandra", "DynamoDB", "Firebase", "Firestore", "Elasticsearch", "Neo4j", "ClickHouse", "TimescaleDB", "MariaDB", "CouchDB", "BigQuery", "Snowflake", "Redshift",
        "AWS", "Azure", "Google Cloud", "GCP", "Docker", "Kubernetes", "K8s", "Terraform", "Ansible", "Jenkins", "GitHub Actions", "GitLab CI/CD", "CircleCI", "ArgoCD", "Helm", "Prometheus", "Grafana", "Datadog", "New Relic", "Splunk", "Nginx", "Apache", "HAProxy", "Istio", "Envoy", "Linux", "Ubuntu", "CentOS", "AWS Lambda", "EC2", "S3", "RDS", "ECS", "EKS", "GKE",
        "Apache Spark", "Kafka", "Airflow", "dbt", "Flink", "Beam", "Luigi", "MLflow", "Kubeflow", "DVC", "Weights & Biases", "W&B", "Databricks", "Delta Lake", "Iceberg", "Hadoop", "Hive", "Presto", "Trino", "Metabase", "Tableau", "Power BI", "Looker", "Superset",
        "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "sklearn", "XGBoost", "LightGBM", "CatBoost", "Hugging Face", "Transformers", "LangChain", "LlamaIndex", "spaCy", "NLTK", "OpenCV", "YOLO", "Stable Diffusion", "Whisper", "Llama", "Mistral", "Gemma", "RAG", "fine-tuning", "prompt engineering", "Pinecone", "Weaviate", "Qdrant", "Chroma",
        "React Native", "Flutter", "Xamarin", "Ionic", "Cordova",
        "Selenium", "Cypress", "Playwright", "Jest", "PyTest", "JUnit", "TestNG", "Mocha", "Chai", "OWASP ZAP", "Burp Suite",
        "Figma", "Sketch", "Adobe XD", "Photoshop", "Illustrator", "InVision", "Principle", "After Effects", "Blender", "Maya", "Unity", "Unreal Engine",
        "Project Management", "Agile", "Scrum", "Kanban", "Jira", "Confluence", "Trello", "Asana", "Stakeholder Management", "Requirements Gathering", "Budgeting", "Forecasting", "Public Speaking", "Technical Writing", "Mentoring", "Cross-functional Collaboration", "Negotiation", "Conflict Resolution", "Change Management",
        "FinTech", "HealthTech", "Cybersecurity", "NLP", "Computer Vision", "Reinforcement Learning", "Time Series Forecasting", "Recommendation Systems", "Search Engines", "Blockchain", "Web3", "Smart Contracts", "Solidity", "DeFi", "Quantitative Finance", "Bioinformatics",
        "AWS Certified Solutions Architect", "Google Professional Cloud Architect", "Microsoft Azure Administrator", "CISSP", "CISM", "PMP", "Scrum Master", "CSM", "Kubernetes Administrator", "CKA"
    ]



    main()
