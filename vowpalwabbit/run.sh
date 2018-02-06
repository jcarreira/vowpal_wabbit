rm -rf /ebs2/click.train.vw.cache
./vw /ebs2/click.train.vw --passes 3 -c --loss_function logistic -P 1000 2>&1 | ts '%.s'
