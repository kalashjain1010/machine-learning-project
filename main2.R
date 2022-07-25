set.seed(42)
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caret) # mark up training and test set categoricals
library(keras) # interface to tensorflow
library(data.table)
train_file<-"sign_mnist_train.csv"
test_file<-"sign_mnist_test.csv"
category<-c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25")

fmnist <- read.csv(train_file)
fmnist2 <- read.csv(test_file)
#mnist <- dataset_mnist()
i_train <- data.matrix(fmnist[,-1]) # strip labels
o_train <- fmnist$label
df<-data.table(fmnist$label)
colnames(df)<-c("X")
train_labels<-df[X == 0, Label := "A"]
train_labels<-df[X == 1, Label := "B"]
train_labels<-df[X == 2, Label := "C"]
train_labels<-df[X == 3, Label := "D"]
train_labels<-df[X == 4, Label := "E"]
train_labels<-df[X == 5, Label := "F"]
train_labels<-df[X == 6, Label := "G"]
train_labels<-df[X == 7, Label := "H"]
train_labels<-df[X == 8, Label := "I"]
train_labels<-df[X == 9, Label := "J"]
train_labels<-df[X == 10, Label := "K"]
train_labels<-df[X == 11, Label := "L"]
train_labels<-df[X == 12, Label := "M"]
train_labels<-df[X == 13, Label := "N"]
train_labels<-df[X == 14, Label := "O"]
train_labels<-df[X == 15, Label := "P"]
train_labels<-df[X == 16, Label := "Q"]
train_labels<-df[X == 17, Label := "R"]
train_labels<-df[X == 18, Label := "S"]
train_labels<-df[X == 19, Label := "T"]
train_labels<-df[X == 20, Label := "U"]
train_labels<-df[X == 21, Label := "V"]
train_labels<-df[X == 22, Label := "W"]
train_labels<-df[X == 23, Label := "X"]
train_labels<-df[X == 24, Label := "Y"]
train_labels<-df[X == 25, Label := "Z"]
#one.hot.labels <- decodeClassLabels(o_train)
i_test <- data.matrix(fmnist2[,-1])
o_test <-fmnist2$label
#one.hot.labels2 <- decodeClassLabels(o_test)

x_train <- i_train
y_train <- o_train
x_test <- i_test
y_test <- o_test

# reshape
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# The y data is an integer vector with values ranging from 0 to 25. 
# To prepare this data for training we one-hot encode the vectors 
# into binary class matrices using the Keras to_categorical() function:
y_train <- to_categorical(y_train, 25)
y_test <- to_categorical(y_test, 25)
# defined labels
#colnames(y_train) = c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24")
colnames(y_train)=c("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y")
#colnames(y_test) = c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25")
colnames(y_test)=c("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y")

#----------------------#
# visualizations
# see individual images
show_letter <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
show_letter2 <- function(arr784, col=gray(255:1/255), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# show a "D"
train_labels$Label[9]
show_letter2(i_train[9,]) 
show_letter(i_train[5,]) 
train_labels$Label[8] # show a "W"
show_letter2(i_train[8,])

# begin layering some models
# for this demo, show the following
# 1. linear dense layers with dropout
# 2. simple convnet with max pooling for translations

# 1. linear stack of layers
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 25, activation = 'softmax')


summary(model)
# compile model with loss, optimizer and metrics defined
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
) 

# training
history <- model %>% fit(
  x_train, y_train, 
  epochs = 1000, 
  batch_size = 128, 
  validation_split = 0.2
)
# testing
model %>% evaluate(x_test, y_test,verbose = 0)
#
#loss  0.3574352
#acc  0.8852
# predict
model %>% predict_classes(x_test)


#-------------------------#
# 2. Trains a simple convnet on the MNIST dataset
# input image dimensions
img_rows <- 28
img_cols <- 28
num_classes <- 25
dim(x_train) <- c(nrow(x_train), img_rows, img_cols, 1) 
dim(x_test) <- c(nrow(x_test), img_rows, img_cols, 1)
input_shape <- c(img_rows, img_cols, 1)
cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# define model
model2 <- keras_model_sequential()
model2 %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', 
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')



summary(model2)
# compile model with loss, optimizer and metrics defined
# compile model
model2 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# training
history <- model2 %>% fit(
  x_train, y_train, 
  epochs = 10,  # 150
  batch_size = 32, # 128
  validation_split = 0.2,
  verbose = 1
)
# testing
model2 %>% evaluate(x_test, y_test,verbose = 0)
scores <- model2 %>% evaluate(
  x_test, y_test, verbose = 0
)

cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
# Test loss: 0.2083899 
# Test accuracy: 0.9293 50 epoch
#
# Test accuracy: 0.9368 150 epoch
# Test loss: 0.2385474

# Test accuracy: 0.9292 10 epoch
# Test loss: 0.219011
# predict
model2 %>% predict_classes(x_test)
