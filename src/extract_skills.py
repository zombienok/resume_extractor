import re
import logging
from typing import List, Tuple, Set, Optional
from functools import lru_cache
import spacy
from sentence_transformers import SentenceTransformer, util

# Configure module-specific logger
logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton-style manager for NLP models with lazy loading."""
    
    _nlp: Optional[spacy.Language] = None
    _sentence_model: Optional[SentenceTransformer] = None
    
    @classmethod
    def get_nlp(cls) -> spacy.Language:
        if cls._nlp is None:
            try:
                cls._nlp = spacy.load("en_core_web_sm")
                logger.info("✓ Loaded spaCy model (en_core_web_sm)")
            except OSError as e:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found. Install with:\n"
                    "python -m spacy download en_core_web_sm"
                ) from e
        return cls._nlp
    
    @classmethod
    def get_sentence_model(cls) -> SentenceTransformer:
        if cls._sentence_model is None:
            cls._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Loaded SentenceTransformer (all-MiniLM-L6-v2)")
        return cls._sentence_model


class SkillExtractor:
    """Extracts skill-like text chunks from raw text using linguistic patterns."""
    
    @staticmethod
    def extract_chunks(text: str) -> List[str]:
        """
        Extract skill-like chunks using:
        - Noun phrases (1-4 words)
        - Cleaned alphanumeric text
        """
        if not text or not text.strip():
            return []
        
        nlp = ModelManager.get_nlp()
        doc = nlp(text)
        chunks = set()
        
        for chunk in doc.noun_chunks:
            cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', chunk.text.strip().replace('\n', ' '))
            words = cleaned.split()
            if 1 <= len(words) <= 4 and len(cleaned) > 2:
                chunks.add(cleaned)
        
        return list(chunks)


class SkillMatcher:
    """Semantic skill matcher using precomputed embeddings."""
    
    def __init__(self, skills_db: List[str], threshold: float = 0.55):
        """
        Initialize matcher with skill database.
        
        Args:
            skills_db: List of canonical skill names
            threshold: Minimum similarity score (0.0-1.0) to consider a match
        """
        if not skills_db:
            raise ValueError("skills_db cannot be empty")
        
        self.skills_db = skills_db
        self.threshold = threshold
        self._skill_embeddings = None
        self._prepare_embeddings()
    
    def _prepare_embeddings(self) -> None:
        """Precompute embeddings for the skill database."""
        model = ModelManager.get_sentence_model()
        self._skill_embeddings = model.encode(
            self.skills_db, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        logger.info(f"✓ Precomputed embeddings for {len(self.skills_db)} skills")
    
    def match(self, text: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Match skills in text against database.
        
        Args:
            text: Resume/job description text
            top_k: Return only top K matches (None = all above threshold)
        
        Returns:
            List of (skill, similarity_score) tuples sorted by score descending
        """
        if not text or not text.strip():
            return []
        
        # Extract candidate chunks
        chunks = SkillExtractor.extract_chunks(text)
        if not chunks:
            logger.debug("No skill-like chunks found in text")
            return []
        
        # Encode chunks
        model = ModelManager.get_sentence_model()
        chunk_embs = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute similarities
        sim_matrix = util.cos_sim(chunk_embs, self._skill_embeddings)
        best_sims = sim_matrix.max(dim=0).values.cpu().numpy()
        
        # Filter and sort matches
        matches = [
            (skill, float(sim)) 
            for skill, sim in zip(self.skills_db, best_sims) 
            if sim > self.threshold
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k] if top_k else matches
    
    def get_similarity(self, text_chunk: str, skill: str) -> float:
        """Get similarity score between a text chunk and specific skill."""
        model = ModelManager.get_sentence_model()
        emb1 = model.encode([text_chunk], convert_to_tensor=True)
        emb2 = model.encode([skill], convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])


# Convenience function for simple use cases
@lru_cache(maxsize=1)
def get_default_matcher() -> SkillMatcher:
    """Get matcher with common tech skills (cached singleton)."""
    common_skills = [
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
    return SkillMatcher(common_skills, threshold=0.60)


# Example usage (when run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    resume = """
    Senior Python Developer with 5+ years building ML systems using TensorFlow 
    and PyTorch. Experienced with AWS cloud infrastructure and Docker containers.
    """
    
    matcher = get_default_matcher()
    results = matcher.match(resume, top_k=5)
    
    print("\nTop skill matches:")
    for skill, score in results:
        print(f"  • {skill}: {score:.3f}")