import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

model.save('digit_recognize.keras')

model=tf.keras.models.load_model('digit_recognize.keras')
loss,accuracy=model.evaluate(x_test,y_test)
print(loss)
print(accuracy)

image_number=1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img=cv2.imread(f"digits/digit{image_number}.png",cv2.IMREAD_GRAYSCALE)[:,:,0]
        cv2.image.resize(img,(28,28))
        img=np.invert(np.array([img]))
        img=img/255.0
        img=np.resize(img,(1,28,28))
        prediction=model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img,cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number+=1