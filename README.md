# HackIA24 - Edge AI System for Smart Cities 
<br>18-19 mai 2024
<br>
<br>
<img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/HackIA24-logo.png width="300">

## Groupe 2 (HackAI_2024_GRP_2)

<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/HackIA24-GRP2-team.png width="300">

<br> *Data Expert* : Francesco Romeo :older_adult: - training, optimisation, explicabilité
<br> *Operations Manager* : Loïc Reboursière :bearded_person: - pruning, compression, benchmark
<br> *Lead Developer* : Rabie Najem :bearded_person: - training, démo
<br> *Designer* : Bérengère Fally :curly_haired_woman: - training, gestion de projet

## Introduction

<br>Ce projet met en œuvre les connaissances acquises en intelligence artificielle durant la formation Hands on AI. Nous avons déployé des modèles de Deep Learning pour des applications en temps réel sur une ressource Edge AI, le Jetson Xavier de NVIDIA. En équipe, nous avons travaillé pendant deux jours pour développer, entraîner et optimiser des réseaux de neurones profonds capables de traiter des séquences vidéo en temps réel. L'objectif était de créer des modules Edge AI pour la reconnaissance faciale et la détection de feu. 
<br>Ce document détaille chaque étape du projet, allant de la conception initiale et l'entraînement des modèles, à leur portage sur ressource Edge, jusqu'à l'optimisation et l'explicabilité des solutions développées. 

## Objectifs
<br>- Développer des réseaux de neurones pour classifier des images, localiser des objets et reconnaître des visages.
<br>- Intégrer les modèles dans un système Edge AI pour des vidéos capturées en temps réel.
<br>- Optimiser la solution pour obtenir un bon compromis entre précision, temps de calcul et espace mémoire.
<br>Tous ces points sont alors intégrés dans une application finale permettant, après déverouillage par reconnaissance faciale, la détection et le tracking du feu. 

## Phases du Projet

### Phase 1 : Développement et entraînement des modèles
Entraînement, validation et test d'un modèle de classification d'images en utilisant PyTorch et Google Colab.

#### Objectifs
- Développer et entraîner des modèles de Deep Learning.
- Utiliser des ressources locales/cloud.

#### Base de données 
<br> Images BD1, BD2 + photos web (kaggle) => 1716 images

<br> fire : 572
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/dataset-fire_category.png width="300">

<br> no-fire : 572
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/dataset-no_fire-category.png width="300">

<br> start-fire : 572
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/dataset-start_fire-category.png width="300">

<br> Dataset final : https://drive.google.com/drive/folders/10FvmIW5iMZp31oEN0X9UfeA_-M2dyt7w?usp=sharing 
 
#### Face recognition
Modèle “Face”
<br> Authentification faciale grâce à la reconnaissance de visages.
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/face_recognition1.png width="300">
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/face_recognition2.jpg width="300">

<br> + Telegram
api ; compte HackIA24 - settings via BotFather (détails : https://docs.google.com/document/d/1P956ckT9Q_z-uEWGx0m6gaDZo564Fe3xZ0dCuvSrkto/edit?usp=drive_link)

#### Modèles développés
- **Modèle 1 : Classification de Feu**
  - **Architecture :** Description de l'architecture.
  - **Données utilisées :** Source et description des données.
  - **Résultats d'entraînement :** Précision, courbes de perte, etc.

- **Modèle 2 : ...

### Phase 2 : Portage sur ressource Edge
Adaptation du modèles développé pour qu'il puisse fonctionner sur Edge, permettant ainsi un traitement en temps réel et une consommation d'énergie optimisée.
#### Configuration de l'environnement Edge
- **Ressources utilisées :** Jetson Xavier.
- **Installation des dépendances :** Liste des dépendances et instructions d'installation.

#### Déploiement des modèles
- **Procédure de déploiement :** Étapes suivies pour déployer les modèles.
- **Challenges rencontrés :** Description des problèmes et solutions trouvées.

### Phase 3 : Optimisation 
#### Optimisation des modèles

<br> !!! tableaux comparaison
métriques des meilleures versions de modèles compressés

### Phase 4 : Compression
- **Pruning :** Description des techniques de pruning appliquées.
Objectif : Réduire le poids pour rendre le modèle plus efficaces en termes de temps de calcul et d'utilisation de la mémoire, sans compromettre significativement leur précision. C'est déployer ces modèles sur des ressources limitées telles que les dispositifs Edge AI.
Résultat : Notre modème étant petit, le pruning sur MobileNet n'a pas vraiment d'effet intéressant, même si compressé par 5 (190.000 paramètres). Nous n'avons que 30% d’accuracy.
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/pruning-MobileNetv3small-parameters.png width="200">
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/pruning-MobileNetv3small-results.png width="200">
<br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/pruning-MobileNetv3small-retrain.png width="200">
Pruning sur MobileNet, pas utile même si compressé par 5 (190.000 paramètres), que 30% d’accuracy.
MobileNet : 4.5 MB | 2.7 MB

- **Quantization :** Description des techniques de quantization appliquées.
Diminuer la taille du modèle et accélérer les calculs sans sacrifier la précision du modèle. Cela permet d'améliorer les performances du modèle sur des appareils avec des capacités de calcul et de mémoire limitées, comme les Jetson Xavier, tout en maintenant une efficacité énergétique élevée.
YOLO V8 : 89.5 MB |23.8 MB

### Phase 5 :  Explicabilité des modèles
- **Framework utilisé :** PyTorch.
- **Méthodes d'explicabilité :** Description des méthodes et résultats obtenus.

### Phase 6 : Benchmark

## Résultats et performance
### Évaluation de la performance
- **Précision :** Résultats de précision avant et après optimisation.
- **Temps d'inference :** Comparaison du temps d'inference avant et après optimisation.
- **Espace mémoire :** Comparaison de l'utilisation de la mémoire avant et après optimisation.

### Analyse des résultats
- **Compromis entre précision et performance :** Description des compromis.
- **Suggestions d'amélioration :** Idées pour améliorer le système.

## Conclusion
### Synthèse du projet
Synthèse des travaux réalisés, des défis relevés et des résultats obtenus.

### Démonstration
https://www.canva.com/design/DAGFneZKn2o/FtJgHbfXzrO6-3KTsr24dQ/edit?utm_content=DAGFneZKn2o&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 

## Environnements, outils utilisés, méthodologie et références utiles
- **Environnements :** Google Colab, Google Colab Pro.
  
- **Frameworks :** TensorFlow, PyTorch, Keras.
  MobileNet, YOLO
- **Outils de collaboration :** 
<br> Drive : https://drive.google.com/drive/folders/1pZg4WNNQ67gFMcpX1OwIWz55K1kt07ip?usp=sharing 
<br> Git : https://github.com/loicreboursiere/HackAI_2024_GRP_2 
<br> Slack : https://hackia-2024.slack.com/
<br> Telegram : https://t.me/HackIA24_bot

- **Méthodologie :**
 <br><img src=https://github.com/loicreboursiere/HackIA_2024_GRP_2/blob/main/img/methodology.jpg width="300">

- **Références :** Liste des références


