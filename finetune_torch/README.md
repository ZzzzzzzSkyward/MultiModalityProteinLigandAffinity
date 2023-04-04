## Dependencies
Please refer to https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_torch#dependencies for environment.

## Finetuning for Cross-Modality Models:
```
mkdir ./weights
```

Finetuning for mask language modeling on concatenation models:
```
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.05
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.15
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.25
```

Finetuning for mask language modeling on cross interaction models:
```
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.05
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.15
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph none --p_seq_mask 0.25
```

Finetuning for graph completion on concatenation models:
```
python main_concatenation_parallel.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.05
python main_concatenation_parallel.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.15
python main_concatenation_parallel.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.25
```

Finetuning for graph completion on cross interaction models:
```
python main_crossInteraction.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.05
python main_crossInteraction.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.15
python main_crossInteraction.py --train 1 --ss_seq none --ss_graph mask --p_graph_mask 0.25
```

Finetuning for joint pre-training on concatenation models:
```
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.05--p_graph_mask 0.05
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.15--p_graph_mask 0.15
python main_concatenation_parallel.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.25--p_graph_mask 0.25
```

Finetuning for joint pre-training on cross interaction models:
```
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.05--p_graph_mask 0.05
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.15--p_graph_mask 0.15
python main_crossInteraction.py --train 1 --ss_seq mask --ss_graph mask --p_seq_mask 0.25--p_graph_mask 0.25
```

You can set or tune the hyper-parameters of regularization terms by adding, ```--l0 ${L0} --l1 ${L1} --l2 ${L2} --l3 ${L3}``` where ```${L0}, ${L1}, ${L2}``` are selected from {0.01, 0.001, 0.0001} and ```${L3}``` from {1, 10, 100, 1000, 10000, 100000}.



ci 100epoch mask mask

train
rmse 0.8125939168608463 pearson 0.9152603066234347 tau 0.7499865762618004 rho 0.9164517342429533
interaction auprc 0.005974394726124555 auroc 0.5820200432707519 binding site auprc 0.04366669731695724 auroc 0.5554693250536341
val
rmse 1.2989839431286971 pearson 0.7313596645458069 tau 0.5224379394536764 rho 0.7209550310093625
interaction auprc 0.006039096612525444 auroc 0.5769448982033176 binding site auprc 0.044061715941709746 auroc 0.5558714434256369
test
rmse 1.461810440080286 pearson 0.6937880090795048 tau 0.4983553667272204 rho 0.6802807408291739
interaction auprc 0.006043211313388521 auroc 0.5862803974075884 binding site auprc 0.04411540858063437 auroc 0.5547811329583584
unseen protein
rmse 1.6616956880206042 pearson 0.5040881785251208 tau 0.31410258537539054 rho 0.4583548813317554
interaction auprc 0.005247369624596104 auroc 0.6161724675396976 binding site auprc 0.03943574752051281 auroc 0.5715483088521632
unseen compound
rmse 1.3549788493326465 pearson 0.7114896786572449 tau 0.5192357009343156 rho 0.7128054377288066
interaction auprc 0.0056730378247868765 auroc 0.5833392405889347 binding site auprc 0.04255063564767807 auroc 0.5588189755288199
unseen both
rmse 1.774136059137434 pearson 0.48789650210490276 tau 0.3424825362704942 rho 0.48825574392014404
interaction auprc 0.005158626750820218 auroc 0.6027914452238465 binding site auprc 0.03836126830450635 auroc 0.5627465484953941





cc mask mask

train
rmse 0.7255791267625292 pearson 0.9342163908119373 tau 0.7815864288692719 rho 0.9347278146810211
interaction auprc 0.004865374732159252 auroc 0.500078432768974 binding site auprc 0.03866155590409124 auroc 0.5000003505052885
val
rmse 1.3088073013359554 pearson 0.7353983838439959 tau 0.5309539370493069 rho 0.7237304344177906
interaction auprc 0.004937257372955947 auroc 0.5000975440515089 binding site auprc 0.03902789284411583 auroc 0.5
test
rmse 1.4293318818776826 pearson 0.70649885243354 tau 0.5089995938842572 rho 0.6909463547452005
interaction auprc 0.004888250187262123 auroc 0.5000628456794698 binding site auprc 0.039265606859090224 auroc 0.5
unseen protein
rmse 1.5336593239280725 pearson 0.5455330732504079 tau 0.35720021229713184 rho 0.5116871784738016
interaction auprc 0.003920612753155597 auroc 0.500080254077957 binding site auprc 0.03382849353924408 auroc 0.5
unseen compound
rmse 1.51242994779753 pearson 0.6338564771609856 tau 0.45375663363553065 rho 0.6410915771677418
interaction auprc 0.004636211852028191 auroc 0.5000601869886377 binding site auprc 0.03762877828399612 auroc 0.5
unseen both
rmse 1.6114725719294667 pearson 0.5518493148680528 tau 0.4018645889302976 rho 0.5650637726381765
interaction auprc 0.003997916005135557 auroc 0.5001305209366824 binding site auprc 0.033450591021771316 auroc 0.5



ci 200epoch 16batch

Namespace(l0=0.01, l1=0.01, l2=0.0001, l3=1000.0, batch_size=32, epoch=200, train=0, data_processed_dir='../data_processed/', ss_seq='mask', ss_graph='mask', p_seq_mask=0.15, p_graph_mask=0.15, p_seq_replace=0.15, p_seq_sub=0.3, p_graph_drop=0.15, p_graph_sub=0.3, supervised=0)
train
rmse 0.8125939168608463 pearson 0.9152603066234347 tau 0.7499865762618004 rho 0.9164517342429533
interaction auprc 0.005974394726124555 auroc 0.5820200432707519 binding site auprc 0.04366669731695724 auroc 0.5554693250536341
val
rmse 1.2989839431286971 pearson 0.7313596645458069 tau 0.5224379394536764 rho 0.7209550310093625
interaction auprc 0.006039096612525444 auroc 0.5769448982033176 binding site auprc 0.044061715941709746 auroc 0.5558714434256369
test
rmse 1.461810440080286 pearson 0.6937880090795048 tau 0.4983553667272204 rho 0.6802807408291739
interaction auprc 0.006043211313388521 auroc 0.5862803974075884 binding site auprc 0.04411540858063437 auroc 0.5547811329583584
unseen protein
rmse 1.6616956880206042 pearson 0.5040881785251208 tau 0.31410258537539054 rho 0.4583548813317554
interaction auprc 0.005247369624596104 auroc 0.6161724675396976 binding site auprc 0.03943574752051281 auroc 0.5715483088521632
unseen compound
rmse 1.3549788493326465 pearson 0.7114896786572449 tau 0.5192357009343156 rho 0.7128054377288066
interaction auprc 0.0056730378247868765 auroc 0.5833392405889347 binding site auprc 0.04255063564767807 auroc 0.5588189755288199
unseen both
rmse 1.774136059137434 pearson 0.48789650210490276 tau 0.3424825362704942 rho 0.48825574392014404
interaction auprc 0.005158626750820218 auroc 0.6027914452238465 binding site auprc 0.03836126830450635 auroc 0.5627465484953941



cc seq=none

train
rmse 0.6585007487898934 pearson 0.9461539207992897 tau 0.8012093763794255 rho 0.9473005740995781
interaction auprc 0.004967966047851648 auroc 0.5090607713194837 binding site auprc 0.038868316874933785 auroc 0.5025927862769433
val
rmse 1.2782486351536222 pearson 0.7513731506438881 tau 0.5479859322405678 rho 0.7464620840740359
interaction auprc 0.0050142759085614825 auroc 0.5053367662721368 binding site auprc 0.03925781150176437 auroc 0.5027045366928226
test
rmse 1.4862694330378305 pearson 0.69849081125657 tau 0.5058534037644858 rho 0.6842936657844705
interaction auprc 0.004987432458648156 auroc 0.5099092997911029 binding site auprc 0.039519932306552374 auroc 0.5030182946500125
unseen protein
rmse 2.0039925073533946 pearson 0.3869893468164879 tau 0.24319103287954155 rho 0.35392671714150825
interaction auprc 0.004019136352097594 auroc 0.5116647677195525 binding site auprc 0.033886460264259885 auroc 0.5009865698196521
unseen compound
rmse 1.4332611129020871 pearson 0.6939590289897523 tau 0.5200043039816151 rho 0.705604704825249
interaction auprc 0.004727879580864118 auroc 0.5097606089746405 binding site auprc 0.037841878705663363 auroc 0.5024795053890683
unseen both
rmse 2.1023341690883677 pearson 0.31377455399553655 tau 0.21316381361766645 rho 0.31667250873820213
interaction auprc 0.004077714982579895 auroc 0.5104196096431997 binding site auprc 0.033531400724450676 auroc 0.5013



cc graph=none

train
rmse 0.8102567299506057 pearson 0.9160571720112828 tau 0.7547591313205577 rho 0.9217097705972663
interaction auprc 0.41497279948045895 auroc 0.856194231958689 binding site auprc 0.5538060219874725 auroc 0.8362084121989496
val
rmse 1.4527395859322558 pearson 0.6891775591578068 tau 0.5028881710602291 rho 0.694347967603868
interaction auprc 0.23549515857650993 auroc 0.8109981832596702 binding site auprc 0.4494813884626949 auroc 0.7990281921676212
test
rmse 1.5294064063338648 pearson 0.6884884632648137 tau 0.4918218551281331 rho 0.6732583837561633
interaction auprc 0.21821125920714704 auroc 0.8049003976927407 binding site auprc 0.4113621644859848 auroc 0.7908431319396221
unseen protein
rmse 1.8256277506902236 pearson 0.43548549796977054 tau 0.23040703195987297 rho 0.33720562751864297
interaction auprc 0.08274912461602257 auroc 0.7881863043047129 binding site auprc 0.2703000695807644 auroc 0.7779948588731573
unseen compound
rmse 1.464264818490387 pearson 0.6894055576728302 tau 0.5067606822435313 rho 0.6943739883276702
interaction auprc 0.21603479985771815 auroc 0.8079622801726589 binding site auprc 0.40848976068179577 auroc 0.7951777763460818
unseen both
rmse 2.047311339769635 pearson 0.36245843289199964 tau 0.2293409841097059 rho 0.3307013358659846
interaction auprc 0.07049942383982401 auroc 0.7796529468136069 binding site auprc 0.2430013993966808 auroc 0.7710998162743372



ci graph=none

Namespace(l0=0.01, l1=0.01, l2=0.0001, l3=1000.0, batch_size=32, epoch=200, train=0, data_processed_dir='../data_processed/', ss_seq='mask', ss_graph='none', p_seq_mask=0.15, p_graph_mask=0.15, p_seq_replace=0.15, p_seq_sub=0.3, p_graph_drop=0.15, p_graph_sub=0.3, supervised=0)
train
rmse 0.701335328591805 pearson 0.9371686206702903 tau 0.783361333669527 rho 0.936641882162551
interaction auprc 0.36854256478463465 auroc 0.8379346309896442 binding site auprc 0.49938758471710326 auroc 0.8145294976956027
val
rmse 1.3763680549667794 pearson 0.7086042018458915 tau 0.5148846198471172 rho 0.7182559974434839
interaction auprc 0.22450878375788152 auroc 0.7969504265967122 binding site auprc 0.4208044507460559 auroc 0.7787473852539359
test
rmse 1.487575220925534 pearson 0.6883301183682193 tau 0.5001466282552655 rho 0.6774223965277544
interaction auprc 0.20330582229463756 auroc 0.7836999667827004 binding site auprc 0.3719922438492989 auroc 0.7679362297017409
unseen protein
rmse 1.7382601687221377 pearson 0.42234734682775643 tau 0.23831217595287896 rho 0.3545562430043014
interaction auprc 0.08427875076406598 auroc 0.7605933032619555 binding site auprc 0.23676750920722464 auroc 0.741645293180654
unseen compound
rmse 1.5080091527223107 pearson 0.6453091076404509 tau 0.46420667891323747 rho 0.650995639663663
interaction auprc 0.1984874175319738 auroc 0.7916544057840802 binding site auprc 0.36967679447597956 auroc 0.7746115781992232
unseen both
rmse 1.8757082905694928 pearson 0.3926416789739031 tau 0.23782413448967785 rho 0.35742752186267795
interaction auprc 0.06942745418171133 auroc 0.7518746065443712 binding site auprc 0.21548520541538296 auroc 0.734060537266295

ci seq=none

Namespace(l0=0.01, l1=0.01, l2=0.0001, l3=1000.0, batch_size=32, epoch=200, train=0, data_processed_dir='../data_processed/', ss_seq='none', ss_graph='mask', p_seq_mask=0.15, p_graph_mask=0.15, p_seq_replace=0.15, p_seq_sub=0.3, p_graph_drop=0.15, p_graph_sub=0.3, supervised=0)
train
rmse 0.636239698323287 pearson 0.9472316995702674 tau 0.8024174466189767 rho 0.9486670301170388
interaction auprc 0.4218137675925103 auroc 0.845558157561799 binding site auprc 0.5360753308598992 auroc 0.8240851719299531
val
rmse 1.3134571919513358 pearson 0.7394160686754844 tau 0.553391739409968 rho 0.7525901710050704
interaction auprc 0.23863861751727836 auroc 0.810838752585741 binding site auprc 0.4353353358030849 auroc 0.8000922841121829
test
rmse 1.5041905378575429 pearson 0.690449973202796 tau 0.49869984009799834 rho 0.6754717417852515
interaction auprc 0.2212126506118836 auroc 0.7953840533920803 binding site auprc 0.4029181928332059 auroc 0.7824433243792757
unseen protein
rmse 1.8895248187349285 pearson 0.3962576053543645 tau 0.2494085618628047 rho 0.3692688635470302
interaction auprc 0.11963747960436848 auroc 0.7881587866411148 binding site auprc 0.33772343600132004 auroc 0.7836150989444274
unseen compound
rmse 1.4999472046804805 pearson 0.6594535851068833 tau 0.4840129882090325 rho 0.672023544710757
interaction auprc 0.21611179252217144 auroc 0.7982297198805687 binding site auprc 0.39835236498395327 auroc 0.7927620620082446
unseen both
rmse 1.739407488509451 pearson 0.48785954650333385 tau 0.3383396023639963 rho 0.5043600308653365
interaction auprc 0.09589940720161594 auroc 0.7710426596000126 binding site auprc 0.29260419899374046 auroc 0.7687005766304665

