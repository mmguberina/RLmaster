# RLmaster
Chalmers master thesis 


### Task 1 - create autoencoder for Atari games
1. start with pong only first
2. create ale-py wrapped in gym environment (so that you can call actions and get new frames), check that you get frames you want
3. wrap generation of frames into some convenient torchvision loader method
4. train the autoencoder on said generation method
5. check results, including nice frame-by-frame comparisons of original and reconstructed frames
--> done. needs more architecture experimenting and tuning, maybe adding so normalization in training, but the main stuff is there.


#### Task 1 - discussion
1. when sampling frames, we necesarily introduce a bias - ideally we would cover the entire
dynamics space of a game so that we learn how to downsample every possible frame.
however, since we need to play the game to reach different parts of the game state space
(which obviously includes the pixel representation of that state space),
we will get biased samples from the state space. 
==========
compare the following 3 scenarios to test this hypothesis and to check how big the differences are:
1. random sample actions to generate frames
2. used an agent trained in a model-free or whatever manner to play the game and collect those frames
--------------
the algorithm to do that is the following:
1. train the latent representation from chosen distribution (generation/sampling method)
2. train a few different algorithms on top of that (say ddq and rainbow)
3. compare results with the same model-free algorithms which are trained end-to-end
4. compare results with some model-based algorithm (say agent51 or something) train in 2 steps and end-to-end

----------------------------------

### Task 2: Training autoencoder in parallel with the policy
-----------------------------------------------
BEFORE PROCEEDING SOME THINGS NEED TO BE PUT IN PLACE FIRST
1. completely automated logging.
rationale: we have no time and no nerves to waste on busy work of manually renaming logging paths and similar bs
1.1. simply save everthing:
1.1.1. save all hyperparameters in a file or something. ensure easy loading and everthing.
1.1.2. save every possibly interesting network, replay buffer, torch tensor, whatever. there are enough terabytes for everything.
1.1.3. automatically generate plots (pull from tensorboards) and save as png. 
1.1.4. generate and save video performance. 
1.1.5. automatically generate a tex file which exctracts the most important elements from all this.
like 3 standard plots + add a line if something was the best or the worst so far. other stuff goes in a table
or something. all this can be a python script which just writes these strings to a file for all i care.
HOW TO KNOW YOU'RE DONE: you change hyperparameters in the experiment script.  you start the training by running
the experiment script.
you're promted to give a textual explanation of why you're doing what you're doing - that gets stored.
you're asked which computer you want to run this on or you let the system select automatically.
the master program prompts the computers it knows about, checks their current load and the number
of tasks they have in their (your) job queue and selects the computer based on that.
you receive a notification when something interesting happens and/or when an experiment is done.

2. setting up autoencoder learning by itself from the replay buffer.
rationale: this will enable easy testing of other embedding models. it is also necessary
to get forward and inverse losses - that needs to be handled by properly sampling from the buffer.
this is easier to deal with individually then by also having to immediatelly intergrate it
with the policy learning. and also it's much easier to debug. it's a boring, but necessary step
1.1. get the most vanilla autoencoder to work - the one which takes one frame, compresses and decompress
it and gets loss computed on it.
1.2. get the forward predicting autoencoder to work. so take in say 4 frames, compress them all into
the 5th frame and calculate loss betweeen this frame prediction and the actual fifth frame that happened.
this will probably work even better if you give it the action that happened betweeen the 4th and 5th frame
as input as well. 
achievign this requires knowing how batches are sampled
from the buffer so that you know that you're actually working with appropriate frames.
and yes this will require a lot of boring index management, but hey, that's life too.
1.3 (LATER STEP) get the inverse model to work too
1.4 write new utility and visualization function to be able to see the results.
this will most likely be easiest to achieve if you just run it in testing mode
from the replay buffer + rendering. that's so because the input will already be processes from 
batches from the replay buffer, so why write new code to the same exact thing?

3. writing the different Autoencoder and latent spaces models as nn.module classes.
these should inherint from a BaseCNNAutoencoder or whatever (call it BaseEmbedder or something).
The differences between them will be minute, if any, and structuring them like this
will make everything a bit easier to take care of.
