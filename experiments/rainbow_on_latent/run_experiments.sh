#!/bin/bash
python atari_rainbow_parallel_training.py --task PongNoFrameskip-v4 --log-name $1
python atari_rainbow_parallel_training.py --task SeaquestNoFrameskip-v4 --log-name $1
python atari_rainbow_parallel_training.py --task MsPacmanNoFrameskip-v4 --log-name $1
python atari_rainbow_parallel_training.py --task BreakoutNoFrameskip-v4 --log-name $1
python atari_rainbow_parallel_training.py --task EnduroNoFrameskip-v4 --log-name $1
python atari_rainbow_parallel_training.py --task QbertNoFrameskip-v4 --log-name $1
python atari_rainbow_parallel_training.py --task SpaceInvadersNoFrameskip-v4 --log-name $1

#python atari_rainbow_parallel_training.py --task PongNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02
#python atari_rainbow_parallel_training.py --task SeaquestNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02
#python atari_rainbow_parallel_training.py --task MsPacmanNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02
#python atari_rainbow_parallel_training.py --task BreakoutNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02
#python atari_rainbow_parallel_training.py --task EnduroNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02
#python atari_rainbow_parallel_training.py --task QbertNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02
#python atari_rainbow_parallel_training.py --task SpaceInvadersNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_02


#python atari_rainbow_parallel_training.py --task PongNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
#python atari_rainbow_parallel_training.py --task SeaquestNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
#python atari_rainbow_parallel_training.py --task MsPacmanNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
#python atari_rainbow_parallel_training.py --task BreakoutNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
#python atari_rainbow_parallel_training.py --task EnduroNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
#python atari_rainbow_parallel_training.py --task QbertNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
#python atari_rainbow_parallel_training.py --task SpaceInvadersNoFrameskip-v4 --log-name rainbow_compression_rec_loss_big_enc_enc_03
