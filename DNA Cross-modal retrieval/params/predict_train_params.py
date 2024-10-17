train_predict_config = dict(
    is_use_gpu = False,
    epochs = 15,
    num_of_seq = 5000,
    len_of_seq = 80,
    percentage = 0.8,
    train_batch_size = 32,
    val_batch_size = 1000,
    save_path = '../model_save/pre_train_predictor_local/train_1',
)