from typing import List, Set, Tuple
from translator import translate_to_english, detect_language



class AllSkills:
    def __init__(self):
        self.skills = set()
        self.embeddings = dict()  # skill → embedding vector
    
    def add_one(self, skill: str, embedding=None):
        self.skills.add(skill)
        if embedding is not None:
            self.embeddings[skill] = embedding
        return
    
    def add_many(self, skills: List[str], embeddings: List[List[float]] = None):
        for i, skill in enumerate(skills):
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            self.add_one(skill, embedding)

    def __contains__(self, skill: str) -> bool:
        return skill in self.skills
    
    def __iter__(self):
        return iter(self.skills)