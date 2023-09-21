import os.path
import time
import pickle
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from src_nowcasting import constants
from src_nowcasting import get_models
from src_nowcasting.sequence_img_generator import *

# get the start time
st = time.time()

##---------------------------- Parameters -----------------------------------##

model_type = 'scnn_csi'                                     # model_type chosen for the training.
study = 'Sequences'                                               # kind of evaluation (single image vs sequence)
loss = 'mae_ghi'                                            # loss for the evaluation.
horizon = 10                                                # time horizon.
no_img = 3
diff = 0
path = constants.REGR_SEQ_DATASET_DIR
img_size = [128, 128]                                       # image size.
img_channels = 1                                            # image channels.
train_batchsize = 32                                        # batch size for train.
val_batchsize = 32                                          # batch size for validation.
test_batchsize = 1                                          # batch size for test.
epochs = 2                                                  # maximum number of epochs.
weight_path = constants.REGR_CHECKPOINTS_DIR          # path to the weight.
log_path = constants.REGR_RESULTS_DIR                       # path to save the CSV file.
checkpoint_path = constants.REGR_CHECKPOINTS_DIR
root_logdir = os.path.join(os.curdir, "run_regression/my_logs")

params1 = {'batch_size': train_batchsize,
           'dim': (img_size[0], img_size[1], 1 * no_img),
           'channel_IMG': img_channels,
           'shuffle': False,
           'iftest': False}

params2 = {'batch_size': train_batchsize,
           'dim': (img_size[0], img_size[1], 1 * no_img),
           'channel_IMG': img_channels,
           'iftest': False}

params3 = {'batch_size': test_batchsize,
           'dim': (img_size[0], img_size[1], 1 * no_img),
           'channel_IMG': img_channels,
           'shuffle': False,
           'iftest': False}
##------------------------------ Train --------------------------------------##

# Retrieve the logs for visualization on TensorBoard.
def get_run_logdir():
    id_run = time.strftime("run_%Y_%m_%d-%H_%M")
    return os.path.join(root_logdir, id_run)


run_id = time.strftime("run_%Y_%m_%d-%H_%M")
run_logdir = get_run_logdir()

print('Preparing Model...')

# Retrieve the CNN structure.
model = get_models.SCNN(input_shape=(img_size[0], img_size[1], no_img+diff))
# Loading the weights of the previous training.
if weight_path is not None:
    print('Loading saved model_type: \'model_{}\'.'.format(model_type))
    model.load_weights(os.path.join(weight_path, f'BestModel_{model_type}_{study}_{loss}_{horizon}.hdf5'))

# Optimizer
optimizer = optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compiling the model_type.
model.compile(optimizer=optimizer,
              loss='mean_absolute_error',        # mean_squared_error
              )

model.summary()

print('Preparing Data...')
sequence_length = 3                     # Number of images in the sequence.
forecast_horizon = 5                    # Minutes to predict.
timestep = 8                            # Minutes between the images in the sequence.
image_type = 'all_GHI'                      # Entire or squared images
# Dataframe loading.
df = pickle.load(open(os.path.join(constants.GHI_DATASET_DIR, f'Dataframe_inv_{image_type}_img_{sequence_length}_ts_{timestep}_{forecast_horizon}.pickle'), 'rb'))
#df = pickle.load(open(r'G:\Il mio Drive\Colab_Notebooks\Dataset\Data\Dataframe_inv_all_GHI_img_3_ts_8_15.pickle', 'rb'))
#df = pickle.load(open(f'G:/Il mio Drive/Colab_Notebooks/Dataset/Data/Dataframe_inv_{horizon}.pickle', 'rb'))

# CSI.
I_ave = sum(df['Target']) / (len(df['Target']))

# train_df, test_df = train_test_split(images, train_size=0.8, shuffle=True, random_state=1)  in case of shuffling
df_tr, df_test = train_test_split(df, train_size=0.8, shuffle=False)
df_train, df_val = train_test_split(df_tr, train_size=0.8, shuffle=False)

# Training generator.
train_generator = DataGeneratorGHI_SCNN(df_train, **params1)

# Validation generator.
validation_generator = DataGeneratorGHI_SCNN(df_val, **params2)

# Test generator.
test_generator = DataGeneratorGHI_SCNN(df_test, **params3)

n_batches = train_generator.__len__()
g, p = train_generator.__getitem__(1)

k=g[9]*255
cv2.imshow('color image', k)
cv2.waitKey(0)

#cv2.imwrite(r'C:\Users\PeoPort\Downloads\3img.png', k)

# Saving data.
callbacks = []
if log_path is not None:
    callbacks.append(CSVLogger(os.path.join(log_path, f'training_model_{model_type}_{study}_{loss}_{horizon}_{run_id}.csv')))
if checkpoint_path is not None:
    callbacks.append(ModelCheckpoint(filepath=os.path.join(checkpoint_path,
                                                           f'BestModel_{model_type}_{study}_{loss}_{horizon}_{run_id}.hdf5'),
                                     verbose=1,
                                     save_best_only=True, save_weights_only=True))
if run_logdir is not None:
    callbacks.append(TensorBoard(run_logdir))

callbacks.append(EarlyStopping(monitor='val_loss',
                               patience=10))

# Train images.
history = model.fit_generator(train_generator,
                              steps_per_epoch=int(df_train.shape[0] / train_batchsize),
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=int(df_val.shape[0] / val_batchsize),
                              callbacks=callbacks
                              )

# Loading data.
#per_rmse = pickle.load(open(f'Dataset/Data/RMSE_persistence_{study}.pickle', 'rb'))        # Google drive
#csghi = pickle.load(open(f'Dataset/Data/csghi_list_{study}.pickle', 'rb'))

per_rmse = pickle.load(open(os.path.join(constants.GHI_DATASET_DIR, f'RMSE_persistence_{study}_inv.pickle'), 'rb'))  # Local
csghi = pickle.load(open(os.path.join(constants.GHI_DATASET_DIR, f'csghi_list_{study}_inv.pickle'), 'rb'))

true_CSI = np.array(df_test['Target'])
true_CSI = np.reshape(true_CSI, (len(true_CSI), 1))
true_GHI = np.multiply(true_CSI, csghi)

a = np.ones(np.shape(true_GHI))*700
diff=true_GHI-a

I_ave = np.sum(true_GHI) / (len(true_GHI))

# Predict.
try:
    model.load_weights(os.path.join(checkpoint_path,
                                    f'BestModel_{model_type}_{study}_{loss}_{horizon}_{run_id}.hdf5'))
    print('Success in loading the best single model')
except:
    print('Fail to load the best single model')
    pass

# Results.
y_hat_CSI = model.predict(test_generator, steps=df_test.shape[0] / test_batchsize)
y_hat_GHI = np.multiply(y_hat_CSI, csghi)

# get the end time.
et = time.time()

# get the execution time.
elapsed_time = time.time() - st
times = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# Errors.
mse = []
mae = []
samples = 0

for i in range(len(y_hat_GHI)):
    mse.append(np.uint64((y_hat_GHI[i] - true_GHI[i]) ** 2))
    mae.append(np.uint64(abs(y_hat_GHI[i] - true_GHI[i])))
    samples += 1

mse_1 = np.sum(mse)
mae_1 = np.sum(mae)
rmse = np.sqrt(mse_1 / len(y_hat_GHI))
mae_error = mae_1 / samples
nrmse = rmse/I_ave*100
print(f'Test RMSE: {rmse}')
print(f'Test nRMSE: {nrmse}')

# forecast skill.
s = (1-rmse/per_rmse)*100

# saving results.
logger = open(f'Regression_Results/metrics_{run_id}', 'w')
logger.write("rmse,nrmse,per_rmse,s, elapsed_time\n")
logger.write(f"{rmse}")
logger.write("{:1}{}".format(",", nrmse))
logger.write("{:1}{}".format(",", per_rmse))
logger.write("{:1}{}".format(",", s))
logger.write("{:1}{}".format(",", times))
logger.close()
