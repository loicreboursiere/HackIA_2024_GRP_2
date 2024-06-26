# HackIA24 - Edge AI System for Smart Cities 
<br>18-19 mai 2024
<br>
<br>
<img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/HackIA24-logo.png width="300">

## Groupe 2 (HackAI_2024_GRP_2)

<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/HackIA24-GRP2-team.jpg width="300">

<br> *Data Expert* : Francesco Romeo :older_adult: - training, optimisation, explicabilité
<br> *Operations Manager* : Loïc Reboursière :bearded_person: - pruning, compression, benchmark
<br> *Lead Developer* : Rabie Najem :bearded_person: - training, démo
<br> *Designer* : Bérengère Fally :curly_haired_woman: - training, gestion de projet

## Introduction

<br>Ce projet met en œuvre les connaissances acquises en intelligence artificielle durant la formation [Hands on AI](https://web.umons.ac.be/fpms/fr/formations/cu-inarti/). Nous avons déployé des modèles de Deep Learning pour la classification et le tracking temps réel de feu ou de départ de feu sur une ressource Edge AI, le Jetson Xavier de NVIDIA. En équipe, nous avons travaillé pendant deux jours pour développer, entraîner et optimiser des réseaux de neurones profonds capables de traiter des séquences vidéo en temps réel sur des ressources limitées. 
<br>L'application finale permet, après déverouillage par reconnaissance faciale, la détection et le tracking du feu.
<br>Ce document détaille chaque étape du projet, allant de la conception initiale et l'entraînement des modèles, à leur portage sur ressource Edge, jusqu'à l'optimisation et l'explicabilité des solutions développées. 
<br> Le développement a été réalisé avec PyTorch utilisé sur des notebooks Jupyter entrainés à partir de Google colab.
 
## Objectifs
<br>- Développer des réseaux de neurones pour classifier des images, localiser des objets et reconnaître des visages.
<br>- Intégrer les modèles dans un système Edge AI pour des vidéos capturées en temps réel.
<br>- Optimiser la solution pour obtenir un bon compromis entre précision, temps de calcul et espace mémoire.

## Phases du Projet

### Phase 1 : Développement et entraînement des modèles

Les codes et les résultats relatifs à cette phase sont repris dans le dossier [training](https://github.com/loicreboursiere/HackIA_2024_GRP_2/tree/main/training).

Les modèles utilisés pour les différentes étapes de ce hackathon peuvent être obtenus en suivant le lien dans le dossier [model](https://github.com/loicreboursiere/HackIA_2024_GRP_2/tree/main/model).

#### Transfert Learning et optimisation des modèles de classification

Plusieurs algorithmes de classification de l'état de l'art ont été testés par Transfert Learning. Une recherche des meilleurs hyperparamètres a été effectuée via une technique de Grid Search. Cette technique vise, pour chaque modèle utilisé, à tester une série de valeurs pour chaque hyperparamètre.

A chaque modification d'un hyperparamètre le modèle est réentrainé et les configurations sont classées les unes par rapport aux autres en fonction de leurs résultats.

Dataset final : https://drive.google.com/drive/folders/10FvmIW5iMZp31oEN0X9UfeA_-M2dyt7w?usp=sharing

#### Base de données 
<br> Images BD1, BD2 du défi 1 + photos web (kaggle) => 1716 images

<br> fire : 572
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/dataset-fire_category.png width="300">

<br> no-fire : 572
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/dataset-no_fire-category.png width="300">

<br> start-fire : 572
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/dataset-start_fire-category.png width="300">


#### Tracking des feux et des départs de feux 

Dès qu'une image est classifiée comme feu ou départ de feu, le modèle [YoloV8](https://github.com/ultralytics/ultralytics) est utilisé pour tracker les feux en temps réel. Celui-ci a été réentrainé pour l'occasion.


### Phase 2 : Portage sur ressource Edge : Jetson Xavier 

#### Reconnaissance de visages pour déverouillage de l'application

L'application présente sur la ressource Jetson Xavier intégrait une reconnaissance faciale qui pouvait être mise à jour par le simple ajout d'une du visage a reconnaître et de la mise à jour du code correspondant à cette photo. Toute l'équipe a donc participé à cette séance photo de mlanière à ce que chacun puisse déverouiller l'application.

<br>Face regognition 1
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/face_recognition1.png width="300">
<br>Face recognition 2
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/face_recognition2.jpg width="300">

Cette application a été modifiée (`EdgeAI/Edge_AI_Module_24_Input-modified.py`) pour intégrer l'envoi d'une notification Telegram lors d'une potentielle intrusion (non reconnaissance du visage).
<br> Cette fonctionnalité a été obtenue par la mise en place d'une API via BotFather et création du compte HackIA24 sur Telegram. Les détails de ce bot sont repris dans le dossier [face_recognition](https://github.com/loicreboursiere/HackIA_2024_GRP_2/tree/main/EdgeAI/face_recognition). 


### Phase 3 : Optimisation et compression de modèles

Plus de détails sur cette phase peuvent être trouvés dans le dossier [compression](https://github.com/loicreboursiere/HackIA_2024_GRP_2/tree/main/compression).

#### Optimisation des modèles

- **Pruning :** 
Objectif : Réduire le poids pour rendre le modèle plus efficaces en termes de temps de calcul et d'utilisation de la mémoire, sans compromettre significativement leur précision. C'est déployer ces modèles sur des ressources limitées telles que les dispositifs Edge AI.

Résultat : Malgré de pauvre résultats (30% d'accuracy) pendant le hackathon, l'entrainement sur 10 périodes après pruning affiche un résultat de 94% sur le validation test. Cela signifie donc que malgré une diminution du nombre de paramètres environ par 5, le modèle semble rester performant. Plus de détails dans le dossier [compression](https://github.com/loicreboursiere/HackIA_2024_GRP_2/tree/main/compression)

MobileNet : 4.5 MB | 2.7 MB

- **Quantization :** 
Objectif : Diminuer la taille du modèle et accélérer les calculs sans sacrifier la précision du modèle. Cela permet d'améliorer les performances du modèle sur des appareils avec des capacités de calcul et de mémoire limitées, comme les Jetson Xavier, tout en maintenant une efficacité énergétique élevée.

Le modèle MobileNet utilisé étant déjà très petit la quantization a été appliquée au modèle YoloV8 avec la variation de taille suivante : 89.5 MB | 23.8 MB

### Phase 4 : Explicabilité des modèles

Pour permettre l'explicabilité de nos modèles, nous utilisons la méthode Grad-CAM (Gradient-weighted Class Activation Mapping) pour localiser les incendies dans les images. Cet algorithme a été utilisé lors de la première épreuve de la certification en IA.

Les résultats de cette phase peuvent être trouvés dans le dossier [ExplainableAI](https://github.com/loicreboursiere/HackIA_2024_GRP_2/tree/main/ExplainableAI).


## Conclusion
### Synthèse du projet
**Principaux résultats :**
<br>
<br> - Modèles de classification et de détection :
<br>Nous avons réussi à développer et entraîner des modèles de classification d'images pour détecter les feux, les départs de feux et les images sans feu avec une précision satisfaisante. L'utilisation de techniques de Transfert Learning et de Grid Search pour l'optimisation des hyperparamètres a permis d'atteindre des performances optimales.
<br>
<br>- Tracking en temps réel :
<br>L'intégration du modèle YOLOv8 pour le tracking en temps réel des feux dans les vidéos de tests a été une réussite, permettant une détection rapide et précise des incidents.
<br>
<br>- Déploiement sur Jetson Xavier :
Nous avons adapté les modèles pour qu'ils fonctionnent efficacement sur le Jetson Xavier, démontrant la faisabilité du traitement en temps réel sur des ressources limitées.
<br>
<br>- Reconnaissance faciale :
<br>L'application inclut une fonctionnalité de reconnaissance faciale pour le déverrouillage, augmentant la sécurité et l'accessibilité de notre solution.
<br>
<br>- Optimisation et compression :
<br>Les techniques de pruning et de quantization ont été appliquées pour réduire la taille des modèles et améliorer leur efficacité sans sacrifier de manière significative la précision. Par exemple, YOLOv8 a été compressé de 89.5 MB à 23.8 MB et le modèle MobileNet_v3_Small a été compressé de 4.5 MB à 2.7 MB.

**Points forts du projet :**
<br>- Travail d'équipe et collaboration :
<br>Le succès de ce projet repose sur une excellente collaboration et répartition des tâches au sein de l'équipe, chacun apportant son expertise spécifique.
<br>
<br>- Utilisation d'outils avancés :
<br>Le choix des outils tels que PyTorch, TensorFlow et Google Colab a permis une gestion efficace du développement et de l'entraînement des modèles.
<br>
<br>- Focus sur l'optimisation :
<br>L'accent mis sur l'optimisation pour le déploiement sur des ressources Edge AI a démontré une approche pratique et applicable à des scénarios réels.

**Perspectives d'amélioration :**
<br>- Amélioration de la précision des modèles :
<br>Il est possible d'améliorer encore la précision des modèles en explorant des architectures plus avancées et en utilisant des ensembles de données plus larges et diversifiés.
<br>
<br>- Robustesse du système :
<br>Renforcer la robustesse du système en intégrant des mécanismes de détection des anomalies et de gestion des erreurs pour assurer une performance stable en conditions réelles.
<br>
<br>- Extensions fonctionnelles : 
<br>Ajouter des fonctionnalités supplémentaires telles que l'alerte en temps réel via des notifications push ou des intégrations avec d'autres systèmes de gestion des incidents pour une réponse plus rapide.
<br>
<br>- Optimisation énergétique : 
<br>Continuer à explorer des techniques pour réduire encore la consommation d'énergie du système.
<br>
<br>- Explicabilité des modèles :
<br>Développer des méthodes d'explicabilité plus avancées pour fournir des insights plus clairs sur le fonctionnement interne des modèles.

## Présentation en équipe à la clôture du HackIA24
https://www.canva.com/design/DAGFneZKn2o/FtJgHbfXzrO6-3KTsr24dQ/edit?utm_content=DAGFneZKn2o&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 

## Appendix : Environnements, outils utilisés et méthodologie
- **Environnements :** Jupyter notebook, Google Colab Pro.
  
- **Frameworks :** PyTorch
  
- **Outils de collaboration :** 
<br> Drive : https://drive.google.com/drive/folders/1pZg4WNNQ67gFMcpX1OwIWz55K1kt07ip?usp=sharing 
<br> Git : https://github.com/loicreboursiere/HackIA_2024_GRP_2 
<br> Slack : https://hackia-2024.slack.com/
<br> Telegram : https://t.me/HackIA24_bot

- **Méthodologie :**
<br> Gestion de projet agile avec une répartition des rôles et tâches entre chaque membre du groupe ainsi qu'une approche par essais/erreurs pour avancer sur les développements.
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/methodology.jpg width="300">
