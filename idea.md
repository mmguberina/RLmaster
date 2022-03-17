the autoencoder is bad on states it hasn't seen.
while that's bad for getting good latent states,
it's great if you're looking for a strong exploration signal!
so a large autoencoder loss can be given as a positive reward to the policy
so as to encourage it to go explore in that direction!
hopefully we get something like sac+ae, but even more efficient (because you find
the best thing a bit faster)

check out burda et al 2018 "exploration by random network distillation"
and see what they did there
