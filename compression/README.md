# Compression résultats

Trois modèles ont été compressés en utilisant des techniques d'élagage (pruning) : MobileNet_V3_small, VGG16 et Yolov8.

## Elagage des modèles MobileNet_v3_small et VGG16

Les deux premiers ont utilisé le script `compression.ipynb` et ont été entrainés à nouveau après l'étape d'élagage. 

L'élagage a utilisé les paramètres suivants via la fonction `progressive_pruning_compression_ratio`: 

| method | speed_up | compression_ratio | global_pruning | iterative_steps | max_sparsity | 	
| :------: | :--------: | :-----------------: | :--------------: | :---------------: | :------------: |
| group_sl | 2 | 4 | TRUE | 400 | 1 |

L'entrainement effectué à la suite de l'élagage a utilisé les paramètres suivants : 

| batchSize | trainSplit | testSplit | validSplit | epochs | criterion | learnRate | optimizer	| imgSize |
| :-------: | :--------: | :-------: | :--------: | :----: | :-------: | :-------: | :--------: | :-----: |
| 32 | 0.8 | 0.1 | 0.1 | 10 | nn.CrossEntropyLoss | 0.01 | Adam | 224 |

Les résultats après élagage sont les suivants : 

| Model	| Params (M) - before | Params (M) - after | MACs (M) - before | MACs (M) - after |	Best val acc (%) |
|:------|:-------------------:|:------------------:|:-----------------:|:----------------:|:----------------:|
| full_model_MobileNet_V3_Small_Weights_test_acc_0.923.pth | 1.08 | 0.27 | 58.66 | 38.48 | 94.15 |
| model_vgg1670.87.pt | 134.27	| 33.09	| 15499.44 | 105.94	| 41.52 |

Nous pouvons voir que les résultats de MobileNet_V3_Small après élagage restent tout à fait corrects alors que ceux de VGG16 chutent drastiquement.

Cela est dû en partie à la taille de MobileNet qui est déjà très limitée par rapport à celle de VGG16.

D'autres modèles de type ResNet150 et ResNet152 ont été testés mais ceux-ci ont subi une modification au moment de leurs sauvegardes, rendant leurs chargements par le script `compression.ipynb`non fonctionnel.
Un nouvel entrainement aurait été nécessaire pour modifier l'architecture de ces modèles et la structure des poids. Nous n'avons pas eu le temps de réaliser ce nouvel entraînement.

## Elagage du modèle YoloV8

L'élagage du modèle YoloV8 a été réalisé grâce au notebook `Yolov8_pruning.ipynb`.

L'élagage utilisé est un élagage non structuré s'appuyant sur la métrique de normalisation L1 avec un facteur de 30% d'élagage. 