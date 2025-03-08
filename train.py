# Create a script called train_models.py
from models import TechnicalSkillsModel, BehavioralSkillsModel
import json
import os

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Sample training data for technical skills
technical_train_data = {
    "questions": [
        "What is the difference between a list and a tuple in Python?",
        "Explain Big O notation",
        "What is a binary search tree?"
    ],
    "answers": [
        "Lists are mutable while tuples are immutable. Lists use square brackets and can be modified after creation, while tuples use parentheses and cannot be changed after creation.",
        "Big O notation is used to describe the performance or complexity of an algorithm. It specifically describes the worst-case scenario and can be used to describe execution time or space used.",
        "A binary search tree is a data structure where each node has at most two children, and all nodes to the left are less than the parent, while all nodes to the right are greater than the parent."
    ],
    "scores": [0.9, 0.8, 0.7]
}

# Sample training data for behavioral skills
behavioral_train_data = {
    "questions": [
        "Tell me about a time you faced a challenging situation",
        "How do you handle disagreements with team members?",
        "What's your approach to learning new technologies?"
    ],
    "answers": [
        "In my last project, we faced a tight deadline. I organized the team, broke down tasks, and we finished on time.",
        "I always try to understand their perspective first. I focus on the issue, not the person, and work towards a solution.",
        "I dedicate time each week to learn something new. I build small projects to practice and join communities to discuss."
    ],
    "scores": [0.85, 0.9, 0.8]
}

# Train technical model
print("Training technical skills model...")
tech_model = TechnicalSkillsModel()
tech_model.train(technical_train_data, epochs=2)  # Reduced epochs for example

# Train behavioral model
print("Training behavioral skills model...")
behav_model = BehavioralSkillsModel()
behav_model.train(behavioral_train_data, epochs=2)  # Reduced epochs for example

print("Training complete!")
