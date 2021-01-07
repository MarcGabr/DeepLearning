import sys
import keras
from matplotlib import pyplot
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

 
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
	trainX = trainX.reshape((60000, 28, 28, 1))
	testX = testX.reshape((10000, 28, 28, 1))
	
	# one hot encode target values
	trainY = to_categorical(trainY,num_classes)  
	testY = to_categorical(testY,num_classes)
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0 
	# return normalized images
	return train_norm, test_norm
  
def define_model(num_classes):
    
    model = Sequential()
            
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.Adam() 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
    
 
# run the test harness for evaluating a model
def run_test_harness(num_classes):
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    
    #make dataset for training 5% and test 0.1% of original
    train_length=int(0.05*60000)
    test_length=int(0.001*10000)
    trainX=trainX[:train_length] 
    testX=testX[:test_length] 
    trainY=trainY[:train_length] 
    testY=testY[:test_length]
    
    print("Size of train dataset used=",train_length)
    print("Size of test dataset used=",test_length)
    
	# define model
    model = define_model(num_classes)
	
    
    #earling stop
    #es2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=0, base_line=0.9 )
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20, min_delta=0.001,restore_best_weights=True)
    #save the best model during validation with best accuracy in the best_model.h5 file
    mc = keras.callbacks.ModelCheckpoint('best_model_fashion_mnist_10.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
       
    
    batch_size = 40
    epochs = 100
    
    model.summary()
    print("\n\nTraining CNN\n\n")
    
    history = model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX,testY),verbose=1,callbacks=[es, mc])	
    
	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    
    
    
   

#===========================================    
    

classes=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

num_classes = len(classes)


print("Type (1) to Trains or (2) to Test the CNN")
opcao=int(input())

if opcao == 1:
 
  # entry point, run the test harness
  run_test_harness(num_classes)

if opcao == 2:
    
    
    
  # load dataset
  trainX, trainY, testX, testY = load_dataset()
  # prepare pixel data
  trainX, testX = prep_pixels(trainX, testX)
    
  #make dataset for training and test 0.1% of original
  train_length=int(0.05*60000)
  test_length=int(0.001*10000)
  trainX=trainX[:train_length] 
  testX=testX[:test_length] 
  trainY=trainY[:train_length] 
  testY=testY[:test_length]  
  
  # load the best model saved
  from keras.models import load_model
  model = load_model('best_model_fashion_mnist_10.h5')
  
  t = 0
  while t >= 0 : 
    print("Type the number of the image to be identified (-1 to exit): ")
    t=int(input())
    
    if t < 0 :
        break
  
    
   

    #predict the class of the selected image
    result = model.predict_classes(testX[t:t+1,::])
  
    #show the predicted class of the image
    print("Class of the image=",classes[result[0]])
    
    import matplotlib.pyplot as plt
    #plot the selected image in the dataset

    text="Selected Image Below - Class of the image=" + classes[result[0]]
    plt.title(text)
    
    plt.imshow(testX[t].reshape((28,28)))
    #plt.imshow(testX[t])
    plt.show()
   
