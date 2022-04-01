#!/bin/bash
rsync -rv --progress -e 'ssh -p 35173' ../RLmaster/ gospodar@chiara4.ddns.net:/home/gospodar/chalmers/MASTER/RLmaster
