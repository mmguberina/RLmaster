#!/bin/bash
rsync -rv -e 'ssh -p 35169' marko@godbowstomath.ddns.net:/home/marko/MASTER/RLmaster/ ~/chalmers/MASTER/RLmaster --exclude="*.git*" --exclude="*buffer.h5*"

