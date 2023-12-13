from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization,Activation,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
img_size=48
batch_size=64
datagen_train=ImageDataGenerator(horizontal_flip=True)
train_generator=datagen_train.flow_from_directory(r"C:\Users\jaswanth\OneDrive\Desktop\plant disease detection\dataset\train",target_size=(48,48),batch_size=batch_size, class_mode='categorical', shuffle=True)
datagen_validation=ImageDataGenerator(horizontal_flip=True)
validation_generator=datagen_train.flow_from_directory(r"C:\Users\jaswanth\OneDrive\Desktop\plant disease detection\dataset\test",target_size=(48,48), batch_size=batch_size,class_mode='categorical', shuffle=True)
#initialising CNN
model=Sequential()
#conv-1
model.add(Conv2D(64,(3,3), padding='same', input_shape= (48,48,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#conv-2
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#conv-3
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#conv-4
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(17, activation='softmax'))
opt=Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
epochs=15
steps_per_epoch=train_generator.n//train_generator.batch_size
validation_steps=validation_generator.n//validation_generator.batch_size
checkpoint=ModelCheckpoint("model_weights.h5", monitor="val_accuracy",save_weights_only=True, model='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience=2,min_lr=1, model= 'auto')
def new_func(train_generator, validation_generator, model, epochs, validation_steps):
    history = model.fit(train_generator, validation_generator, model, epochs, validation_steps)
model.save('my_disease.h5')
from keras.models import load_model
classifier = load_model('my_disease.h5')
classifier = model.save ('saved_final_model/my_model')