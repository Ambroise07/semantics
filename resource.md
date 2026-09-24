. ChatGPT dans la littérature scientifique

Il n'y a toujours pas de bon livre sur ChatGPT, la meilleure façon de mieux comprendre est de lire cette collection d'articles scientifiques dans l'ordre chronologique:

1. Bengio et al, 2003, A Neural Probabilistic Language Model
Il montre une matrice partagée C sur les jetons d'entrée pour projeter les jetons dans un espace à ~ 30 dimensions d'une manière similaire à word2vec. Ensuite, il y a un calcul avec un vecteur d'anticipation et une couche softmax qui calcule la probabilité du jeton suivant, qui est choisi via un processus d'échantillonnage de distribution multinomiale.

2. Bengio et al, 2015, Neural machine translation by jointly learning to align and translate
Il dispose d'un encodeur-décodeur de réseau neuronal récursif pour traduire d'une langue à l'autre

3. OpenAI Team, 2017, Attention is all you need
Il remplace le réseau neuronal récursif de [2] par un nouveau composant de réseau neuronal appelé le transformateur. Il décrit en détail le processus d'attention, comment il peut être fait en parallèle avec l'attention multi-tête, et la nécessité de masquer le mécanisme d'attention sur la partie décodeur, afin que les informations des jetons précédents ne circulent que vers les jetons suivants et non l'inverse. . Il décrit comment superposer des plongements positionnels aux jetons.

4. Alec Redford and OpenAI Team, 2018, Improving language understanding by Generative Pre Training
La partie codeur de [3] est abandonnée, pour passer d'un modèle pour faire des traductions à un modèle qui génère la probabilité pour le prochain jeton. GPT-1 est né. Il s'agit d'un décodeur à 12 couches avec 12 têtes d'auto-attention masquées utilisant 768 dimensions. Il a 117 millions de paramètres. Il introduit le concept de réglage fin supervisé pour adapter le réseau à des tâches telles que l'inférence du langage naturel, la réponse aux questions, la similarité sémantique et la classification.

5. Alec Redford and OpenAI Team, 2019, Language Models are Unsupervised Multitask Learners
Ce document présente GPT-2 et montre ses capacités. Il contient 1,5 milliard de paramètres. On observe que les grands réseaux sont plus aptes aux tâches décrites dans [4].

6. OpenAI Team, julliet 2022, Language Models are Few Shot Learners
Cet article décrit l'évolution de GPT-2 à GPT-3 qui compte désormais 175 milliards de paramètres. 96 couches avec 96 têtes avec une dimensionnalité de tête de 128 produisent ce que nous appelons ChatGPT, actuellement disponible sur https://chat.openai.com/.

GPT3 a une fenêtre de contexte de 2048 jetons (~ 1500 mots). Avec GPT-3.5, la fenêtre de contexte a été augmentée à 4096 (3000 mots de texte anglais).

Le nombre total d'opérations pour calculer les paramètres du réséau est donné sous la forme de flops 3.14E23. A partir de ce nombre, en supposant des GPU de 10 Teraflops, nous estimons que l'entraînement du GPT-3 a nécessité un cluster d'environ 4'000 GPU calculant à plein temps pendant 3 mois. Il introduit également le concept selon lequel en donnant 1 à 3 exemples (plans) au modèle comme contexte, le modèle se comporte mieux.

7. OpenAI Team, 2023, GPT-4 Technical Report
GPT-4 a la fenêtre de contexte augmentée à 8192 jetons et a la capacité de recevoir des images en entrée. Le document décrit ce que le modèle peut faire et comment il se comporte sur plusieurs tests courants, mais ne divulgue aucun chiffre sur la taille du modèle ou le temps de formation nécessaire pour le calculer.

8. Une autre source intéressante pour en savoir plus sur ChatGPT est : il suffit de demander à ChatGPT 3.5 comment cela fonctionne ! Un compte Google est nécessaire pour se connecter, mais c'est gratuit. Au mieux, posez des questions sur les "chatbots complexes" car l'utilisation de "ChatGPT" dans la question déclenche une sorte de réponse standard.

3. Exécutez ChatGPT sur votre ordinateur local

Le meilleur projet Open Source à suivre est GPT4All. Il s'agit d'un chatbot complet qui s'exécute localement sur votre ordinateur où il est possible de sélectionner parmi différents modèles entre 4 milliards et 13 milliards de paramètres. La RAM requise est de 16 Go pour les modèles plus grands avec 13 milliards de paramètres. Des modèles plus petits mais plus rapides pour générer des réponses avec 7 milliards de paramètres ne nécessitent que 8 Go de RAM. Il existe des installateurs pour Windows, Linux et Mac OS X. Chaque semaine, 
il y a des mises à jour du logiciel ou des modèles disponibles.
