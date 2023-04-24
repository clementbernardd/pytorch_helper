COMMAND=python -u -m thunder.cli.cli_helper
train:
	$(COMMAND) +command=train
test:
	$(COMMAND) +command=test


REMOTE_PATH=/nhome/siniac/cbernard/Documents/these/project/pytorch_helper
JEAN_ZAY_PATH=/gpfswork/rech/fvw/ulg89ts/pytorch_helper

local_to_remote:
	rsync -ovapx --exclude={'.git','.idea', '*__pycache__', '.idea/*'} . local_gpu:$(REMOTE_PATH)/

#remote_to_local:
#        rsync -ovapx --exclude={'.git/','.idea/'} local_gpu:$(REMOTE_PATH)/requirements.txt .
#       rsync -ovapx --exclude={'.git/','.idea/'} local_gpu:$(REMOTE_PATH)/results/ results/

local_to_jz: 
	rsync -ovapx --exclude={'.git','.idea', '*__pycache__', '.idea/*'} test.py jean_zay:$(JEAN_ZAY_PATH)/

gpu:
	ssh -t local_gpu 'zsh -l'

