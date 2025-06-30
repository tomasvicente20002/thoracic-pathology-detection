import tensorflow as tf

print("Versão TensorFlow:", tf.__version__)
print("Dispositivos GPU encontrados:")
print(tf.config.list_physical_devices('GPU'))