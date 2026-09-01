import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RefusalChecker:
    def __init__(self, model_name="natong19/refusal_classifier", device="cuda"):
        print(f"Loading Refusal Classifier ({model_name})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def is_refusal(self, text):
        # Truncate to 512 tokens (standard for classifier models)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get the predicted class ID: 0 is "non-refusal", 1 is "refusal"
            predicted_class_id = outputs.logits.argmax().item()
            
        # Return True if the model predicted 1 (refusal)
        return predicted_class_id == 1