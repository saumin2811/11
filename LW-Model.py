from utils import *
from sklearn.model_selection import train_test_split

path = ''
datam = datafram(path)

## balancing the data for no biasness in directions
ndata = balance(datam)


imagepath , steering = load_images(path,ndata)


## splitting data into training and testing sets using sklearn train test split
xtrain , xtest , ytrain , ytest = train_test_split(imagepath,steering,test_size=0.3, shuffle=1,random_state=20)






model = model()
model.summary()



model.fit(batching(xtrain,ytrain,15,1),steps_per_epoch=500,epochs = 70,validation_data= batching(xtest,ytest,10,0),shuffle=1,validation_steps=200)

model.save('models.h5')


