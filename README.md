# ⚙️ Cogitator — Servitor AI Voice Assistant

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com/)
[![Adeptus Mechanicus Approved](https://img.shields.io/badge/Omnissiah-Sanctified-red.svg)](#)

Un assistant vocal local et temps réel inspiré de l'univers **Warhammer 40,000 (Adeptus Mechanicus)**. Le système écoute l'utilisateur, interprète la requête via un LLM enrichi avec du vocabulaire spécifique, et répond avec une synthèse vocale modifiée en temps réel pour imiter la voix d'un **Serviteur / Cogitateur**.

---

## 📋 Table des matières

1. [Objectif du projet](#-objectif-du-projet)
2. [Pré-requis](#-pré-requis)
3. [Installation](#-installation)
4. [Dépendances](#-dépendances)
5. [Structure du projet](#-structure-du-projet)
6. [Explication du code](#-explication-du-code)
7. [Utilisation](#-utilisation)

---

## 🎯 Objectif du projet

L'objectif de **Cogitator** est de fournir un agent vocal complet, autonome et fonctionnant **100 % en local** sur PC. 

### Fonctions clés :
* **Détection Vocale Dynamique (VAD) :** Analyse le flux audio du microphone via Silero VAD pour démarrer l'enregistrement à la parole et l'interrompre automatiquement dès qu'un silence est détecté.
* **Reconnaissance Vocale (STT) :** Transcrit la parole de l'utilisateur instantanément via `faster-whisper`.
* **Cerveau Cognitif (LLM) :** Utilise un modèle LLM léger sous Ollama (`qwen2.5:3b` personnalisé en `cogitator`), configuré pour répondre de manière concise, rigide et rituelle.
* **Moteur RAG (Lexique 40k) :** Intercepte les requêtes pour injecter du contexte et du vocabulaire issu de l'univers Warhammer 40k stocké dans une base vectorielle locale (`ChromaDB`).
* **Synthèse et Traitement Audio (TTS + DSP) :** Génère la voix via `sherpa-onnx` et applique un traitement numérique du signal (DSP) via `Pedalboard` et `NumPy` (Ring Modulation, Bitcrush, GSM Compressor, PitchShift) pour obtenir un rendu vocal métallique et synthétique.

---

## 🛠️ Pré-requis

Avant de procéder à l'installation, assure-toi d'avoir configuré sur ta machine :

* **Système d'exploitation :** Windows 10/11 (ou Linux)
* **Python :** Version **3.10** recommandée (ou 3.11)
* **Ollama :** Téléchargé et installé depuis [ollama.com](https://ollama.com/).
* **Matériel :**
  * Un microphone actif.
  * Des enceintes ou un casque.
  * *(Optionnel)* Un GPU NVIDIA avec CUDA pour accélérer la transcription Whisper et la réponse du LLM.

---

## 🚀 Installation

### 1. Cloner le projet et se placer dans le répertoire
```powershell
git clone <URL_DU_REPO>
cd Cogitator
```

### 2. Créer et activer l'environnement virtuel Python

```powershell
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
```

### 3. Installer les dépendances Python

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Télécharger et placer les modèles vocaux Sherpa-ONNX
Crée le dossier `models/` à la racine du projet et déposes-y les fichiers de ton modèle vocal Piper/Sherpa-ONNX :

```powershell
mkdir models
```

Assure-toi de placer dans ce dossier `models/` :

* Le fichier du modèle acoustic `.onnx` (ex: `en_US-alan-medium.onnx` ou équivalent français).
* Le fichier `tokens.txt` correspondant.

Exemple complet d'ajout de modèle vocal :
```powershell
# 1. Se placer dans le dossier models/
cd models

# 2. Créer le sous-dossier de la nouvelle voix (ex: voix française upmc)
mkdir fr_upmc
cd fr_upmc

# 3. Télécharger le modèle Piper FR via curl
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-upmc-medium.tar.bz2

# 4. Extraire l'archive bz2/tar
tar -xvf vits-piper-fr_FR-upmc-medium.tar.bz2 --strip-components=1

# 5. Télécharger la base phonétique espeak-ng-data (nécessaire si absente)
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
tar -xvf espeak-ng-data.tar.bz2

# 6. Nettoyer les archives téléchargées
Remove-Item *.tar.bz2
```

### 5. Configurer et créer le modèle Ollama (`cogitator`)

Assure-toi qu'Ollama est lancé en arrière-plan, puis tire le modèle de base et compile le modèle personnalisé à partir du `Modelfile` :

```powershell
# Télécharger le modèle de base
ollama pull qwen2.5:3b

# Créer le modèle personnalisé "cogitator"
ollama create cogitator -f Modelfile
```

Pour ajouter d'autres modèles de langage (LLM) utilisables par Ollama :
```powershell
# Télécharger un modèle de taille moyenne (ex: Mistral 7B)
ollama pull mistral

# Ou un modèle spécialisé en code / logique
ollama pull codellama:7b

# Lister tous les modèles installés localement
ollama list
```
**Ne pas oublier de modifier le fichier ``Modelfile`` quand vous voulez changer de modèle utilisé par Ollama**, exemple : 
```dockerfile
FROM mistral:latest

... # Le reste du fichier est inchangé
```

---

## 📦 Dépendances

Le fichier `requirements.txt` regroupe l'ensemble des modules Python nécessaires :

```text
# --- Acquisition et Synthèse Vocale (STT / TTS) ---
faster-whisper
sherpa-onnx

# --- Traitement Audio et Matrice Audio ---
sounddevice
numpy
scipy
pedalboard

# --- Base de Données Vectorielle (RAG Lexique 40k) ---
chromadb

# --- Communication API ---
requests

```
Ainsi que tous les autres packages liés.

---

## 📂 Structure du Projet

```text
Cogitator/
│
├── chroma_db/               # Base de données vectorielle locale générée par ChromaDB
├── models/                  # Modèles de synthèse vocale organisés par sous-dossiers
│   ├── en_alan/             # Modèle anglais (ONNX, tokens, espeak-ng-data) (à télécharger vous même)
│   └── fr_upmc/             # Modèle français (à télécharger vous même)
├── venv/                    # Environnement virtuel Python
├── config.py                # Fichier central de configuration (constantes, voix, VAD, DSP)
├── Modelfile                # Configuration du Prompt Système et des paramètres Ollama
├── warhammer_lexicon.txt    # Fichier source contenant le vocabulaire et le lore 40k
├── rag_engine.py            # Module RAG (Indexation et recherche vectorielle)
├── servitor.py              # Pipeline principal (VAD -> STT -> RAG -> Ollama -> TTS -> DSP)
├── requirements.txt         # Dépendances du projet
└── README.md                # Documentation
```

---

## 🔍 Explication du Code

### 1. `config.py`
Centralise l'intégralité des **constantes et paramètres du système** :
* **Modèle vocal actif :** Choix du sous-dossier dans `models/` via `ACTIVE_MODEL_FOLDER`.
* **Réglages VAD & STT :** Seuil de probabilité, durée de silence (`VAD_SILENCE_DURATION`), taille du modèle Whisper (`WHISPER_MODEL_SIZE`) et prompt d'orientation phonétique (`STT_INITIAL_PROMPT`).
* **Traitements Audio DSP :** Fréquence de Ring Modulation, hauteur (`DSP_PITCH_SHIFT_SEMITONES`), filtres passe-haut/bas et niveau de Bitcrush.
* **LLM & RAG :** URL Ollama, timeout et nombre de documents extraits de ChromaDB.

### 2. `Modelfile`
Définit la personnalité du Serviteur sous Ollama. Il fixe la température à `0.2` pour limiter les hallucinations, restreint le nombre de tokens générés (`num_predict`), et impose un ton froid et rituel. Il ordonne au modèle de **demander des clarifications** si la requête est incomplète.

### 3. `rag_engine.py` (LexiconRAG)
* **Classe `LexiconRAG` :** Initialise une base de données `ChromaDB` persistante basée sur le chemin configuré.
* **`load_lexicon()` :** Lit `warhammer_lexicon.txt`, découpe les entrées et génère les embeddings vectoriels.
* **`search(query)` :** Récupère les entrées de vocabulaire les plus pertinentes pour enrichir le prompt LLM.

### 4. `servitor.py` (Pipeline Principal)
* **`record_audio_with_vad()` :** Écoute le flux du microphone en continu via `sounddevice.InputStream` et évalue la probabilité de parole en temps réel avec `Silero VAD`. L'enregistrement s'arrête automatiquement après une période de silence configurée.
* **`transcribe_audio()` :** Transcrit le fichier temporaire WAV via `faster-whisper` en exploitant `STT_INITIAL_PROMPT` pour garantir la précision des termes techniques et commandes d'arrêt.
* **`query_llm()` :** Interroge le moteur RAG puis transmet la requête enrichie à l'API locale Ollama (`http://localhost:11434/api/generate`).
* **`speak_servitor()` :** Synthétise le texte via `sherpa-onnx`, applique une **Ring Modulation** sinusoïdale en NumPy pour la monotonie robotique, puis passe le signal dans la chaîne d'effets `Pedalboard` (`PitchShift`, `Bitcrush`, `GSMCompressor`, `Distortion`, `Filters`) avant la lecture.

### 4. Dossier `models/`
Contient l'ensemble des poids acoustiques locaux exploités par le moteur **Sherpa-ONNX** pour la synthèse vocale hors-ligne. Cela garantit une génération vocale ultra-rapide (< 200 ms) sans dépendre d'un service cloud.

---

## Reste à faire
- Ajout d'une voix française
- Modification de la voix pour avoir quelque chose de ressemblant à [cette vidéo](https://www.youtube.com/watch?v=nmeKwb7KlRM)
- Faire une interface pour modifier la voix et afficher les texte en input et en output (streamlit / gradio)

---
*Ave Omnissiah. May the Machine Spirit guide your execution.*