#!/bin/bash
#rsync -rv --progress -e 'ssh -p 35173' /home/gospodar/chalmers/MASTER/RLmaster/ gospodar@chiara4.ddns.net:/home/gospodar/chalmers/MASTER/RLmaster
rsync -rv -e 'ssh -p 35173' \
		--exclude='*dataset' \
		../RLmaster/ \
		gospodar@chiara4.ddns.net:/home/gospodar/chalmers/MASTER/RLmaster
		#gospodar@machine:/home/gospodar/chalmers/MASTER/RLmaster
