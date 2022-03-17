train an autoencoder on collected images.
do that by collecting samples in an online fashion, storing them in a buffer,
and using minibatches from that.
play around with it until you get decent results.

the easiest way to incorporate a pre-trained ae into a model
is to write a class which inherits from gym.Wrapper. 
then you can use the encoder to downsample observations
and whatever uses those observations will #justwork
