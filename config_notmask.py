config = dict()

config['resume_path'] = 'Model_training_checkpoints/model_resnet34_cheahom_triplet_epoch_20_roc0.9337.pt'

config['model'] = 34 # 18 34 50 101 152
config['optimizer'] = 'adagrad'      # sgd\adagrad\rmsprop\adam
config['predicter_path'] = 'shape_predictor_68_face_landmarks.dat'

config['Learning_rate'] = 0.00001
config['image_size'] = 256        # inceptionresnetv2————299
config['epochs'] = 190         

config['train_batch_size'] = 5#136
config['test_batch_size'] = 5

config['margin'] = 0.5
config['embedding_dim'] = 128
config['pretrained'] = False
config['save_last_model'] = True
config['num_train_triplets'] = 12500
config['num_workers'] = 6


config['train_data_path'] = 'Datasets/train_face_notmask'
config['mask_data_path'] = 'Datasets/train_mask_notmask'
config['train_data_index'] = 'train_face_notmask.csv'
config['train_triplets_path'] = 'Datasets/training_triplets_' + str(config['num_train_triplets']) + 'notmask.npy'
config['test_pairs_paths'] = 'Datasets/test_pairs.npy'
config['LFW_data_path'] = 'Datasets/lfw_funneled'
config['LFW_pairs'] = 'Datasets/pairs.txt'
