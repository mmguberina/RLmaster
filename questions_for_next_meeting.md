## 1. Results
-------------
Please comment on the supposed reasons why it's worse:
1. the q-network is smaller
2. the learned features are so much worse then the ones
the q-network learns that it's actually faster to learn them
than to learn what these mean
3. the autoencoder was train on per-image basis, while the 
q-network's feature layers took in the stacked 4 frames

2. Bug discussion
-----------------
1. What's the best way to manipulate the computation graph so that
the q gradients are not passed through the encoder?
2. On that note, since we're training both at the same time,
would it work if we passed both gradients through the encoder?
that's what yarats does in sac+ae paper (altough the algorithm
is a policy-gradient one, maybe that plays a role)

3. Ideas for making this better
---------------------------------
1. Making the autoencoder take in stacked frames (the code for this is ready)
2. Making the autoencoder predict the next frame (some timesteps in the future possibly)
3. Training an inverse dynamics model (s_{t}, s_{t+1}) -> a_t in a self-supervised manner
and use some layers of that to construct the features.
this is much closer to the features that the q-network neess (arguments from icm paper)
4. using a forward dynamics model trained on features above and
then using that for q-learning?


============
1. try having cnn layers in q-net
2. frame predicting
3. try passing the both q and reconstruction loss through the encoder
4. renormalize the bottleneck layer
