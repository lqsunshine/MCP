@REM Dataset:fvc Environment: cross_hash
@REM fvc
@REM clean
python main.py train --flag 'fvc' --group 'clean'
python main.py test --flag 'fvc' --group 'clean'

@REM lbl
python backdoor_main.py train --flag 'fvc' --backdoor True --group 'StegaStamp_label' --backdoor_trigger 'StegaStamp'
python backdoor_main.py test --flag 'fvc' --group 'StegaStamp_label' --backdoor_trigger 'StegaStamp'

@REM mcp
python backdoor_main_lg.py train --flag 'fvc' --backdoor_loss True --group 'StegaStamp_loss_lg' --backdoor_trigger 'StegaStamp'
python backdoor_main_lg.py test --flag 'fvc' --group 'StegaStamp_loss_lg' --backdoor_trigger 'StegaStamp'