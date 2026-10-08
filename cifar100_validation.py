"""Controles operativos de aprendizaje; no imponen una forma a la curva de replay."""

# Claves que deben coincidir entre la ejecución y el checkpoint reutilizado, con
# el valor que se supone cuando un manifiesto antiguo no las registra. Los
# checkpoints antiguos se entrenaron sin normalización ni aumento de datos.
CONTRACT_DEFAULTS = {
    "num_clients": None,
    "seed": None,
    "dataset_revision": None,
    "train_samples_per_group": None,
    "sampling": None,
    "normalize": False,
    "augment": False,
}


def validate_training_contract(config, checkpoint_config):
    """El replay debe conservar una fracción del mismo conjunto aprendido en A, con el mismo preprocesado."""
    differences = [key for key, default in CONTRACT_DEFAULTS.items()
                   if config.get(key, default) != checkpoint_config.get(key, default)]
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
