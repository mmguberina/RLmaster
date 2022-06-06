#!/bin/bash
rsync -rv -e 'ssh -p 35173' ~/chalmers/MASTER/RLmaster/ marko@godbowstomath.ddns.net:/home/marko/MASTER/RLmaster --exclude="*.git*" --exclude="*buffer.h5*" --exclude="*log*"

