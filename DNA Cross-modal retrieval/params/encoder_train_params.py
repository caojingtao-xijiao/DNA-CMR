train_encoder_config = dict(
train_data_path = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/table_train_data.h5',
val_data_path = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/table_val_data.h5',
encoder_train_batch_size = 1000,
encoder_val_batch_size = 5000,
predictor_data_num = 1000,
predictor_batch_size = 32,
predict_params_path ='../model_save/pre_train_predictor_local/train_1/predictor_trained.pth',
is_use_gpu = True,
epochs = 20,
refit_every = 1,
refit_epochs = 10,
save_path = '../model_save/encoder_train/train_3',
)