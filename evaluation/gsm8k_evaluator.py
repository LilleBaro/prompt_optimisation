"""
  prompt 
    |
  model
    |
  reponse
    |
extraction reponse
    |
comparaison gmk8s
    |
accuracy

accuracy = evaluation.evaluate(prompt, dataset)
"""

class GSM8KEvaluator:
    def evaluate(self, prompt, dataset):
        ...