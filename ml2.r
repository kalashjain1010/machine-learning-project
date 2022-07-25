library(tensorflow)
library(keras)
library(stringr)
library(readr)
library(purrr)


cifar <- dataset_cifar10()

index <- 1:30

class_names <- c('airplane' , 'automobile', 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck')

par(mfcol = c(5,6),mar= rep(1,4) , oma = rep(0.2,4))

cifar$train$x[index,,,] %>%
  array_tree(1) %>%
  set_names(class_names[cifar$train$y[index]+1]) %>%
  map(as.raster,max = 255) %>%
  iwalk(~{plot(.x); title(.y)})


#model
model1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32 , kernel_size = c(3,3) , activation = "relu" , input_shape = c(32,32,3)) %>%
  layer_conv_2d(filters = 64 , kernel_size = c(3,3) , activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64 , kernel_size = c(3,3) , activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2))

summary(model1)


model1 %>%
  layer_flatten() %>%
  layer_dense(units = 256 , activation = "relu",input_shape = c(32,32,3)) %>%
  layer_dense(units = 128 , activation = "relu") %>%
  layer_dense(units = 64 , activation = "relu") %>%
  layer_dense(units = 10 , activation = "softmax")

summary(model1)

model1 %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)
summary(model1)

history <- model1 %>%
  fit(
    x = cifar$train$x , y = cifar$train$y,
    epochs = 10,
    validation_split = 0.2,
    use_multiprocessing = TRUE
  )