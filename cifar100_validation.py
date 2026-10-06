"""Controles operativos de aprendizaje; no imponen una forma a la curva de replay."""


def validate_training_contract(config, checkpoint_config):
    """El replay debe conservar una fracción del mismo conjunto aprendido en A."""
    keys = ("num_clients", "seed", "dataset_revision", "train_samples_per_group", "sampling")
    differences = [key for key in keys if config.get(key) != checkpoint_config.get(key)]
    if differences:
        raise ValueError("El checkpoint y el conjunto A para replay no coinciden en: " + ", ".join(differences))


def learning_problem(test_b, training_b, replay_training, config):
    if test_b < config["min_new_task_accuracy_percent"]:
        return "Aprendizaje insuficiente en el test de B"
    if training_b < config["min_training_accuracy_percent"]:
        return "Aprendizaje insuficiente en las propias imágenes de entrenamiento B"
    if replay_training is not None and replay_training < config["min_training_accuracy_percent"]:
        return "El modelo no aprende suficientemente las propias imágenes del replay"
    return None
