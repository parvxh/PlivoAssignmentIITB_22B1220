from transformers import AutoModelForTokenClassification

def create_model(model_name_or_dir, num_labels=15):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_dir,
        num_labels=num_labels
    )
    return model
