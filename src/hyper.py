search_space = {
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
    'weight_decay': hp.uniform('weight_decay', 0.0001, 0.1),
    'momentum': hp.uniform('momentum', 0.1, 0.9),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
    'num_epochs': hp.choice('num_epochs', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    'cv_fold': hp.choice('cv_fold', [0, 1, 2, 3, 4]),
}