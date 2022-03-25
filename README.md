# RLmaster
Chalmers master thesis 


### Task 1 - create autoencoder for Atari games
1. start with pong only first
2. create ale-py wrapped in gym environment (so that you can call actions and get new frames), check that you get frames you want
3. wrap generation of frames into some convenient torchvision loader method
4. train the autoencoder on said generation method
5. check results, including nice frame-by-frame comparisons of original and reconstructed frames


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
3. hack the game so that you generate truly uniformly random states and sample frames from that
--------------
the algorithm to do that is the following:
1. train the latent representation from chosen distribution (generation/sampling method)
2. train a few different algorithms on top of that (say ddq and rainbow)
3. compare results with the same model-free algorithms which are trained end-to-end
4. compare results with some model-based algorithm (say agent51 or something) train in 2 steps and end-to-end

----------------------------------


Yes, that is actually a valid problem. Store the frames from a random policy in a large buffer, and then train autoencoder on mini-batches in this buffer. It should give similar effect to hashing & removing frames. Maybe also only store every Nth frame. Better idea is to run multiple agents in parallel, and store each of their experience in a single buffer, and use that to train AE (this is what asynchronous A3C does).


Paper-2 is already what I told you before, that directly use AE, instead of VAE, so their suggestions about tweaking learning rate will not help much.


3rd way is to train a single AE, using multiple agents run not only on a single game, but variety of games (don't worry about sample efficiency for now). Because I believe everything your trying now won't work, since you are not able to reach many new states using a random policy, since the game ends quite soon. This is exactly the problem tacked by this paper: https://arxiv.org/abs/1901.10995 , and you should rather focus on its extensions or newer papers citing & using this idea.



Best,
Divya

