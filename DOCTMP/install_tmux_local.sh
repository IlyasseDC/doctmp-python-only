#!/bin/bash

# Configuration
TMUX_VERSION=3.3a
PREFIX=$HOME/.local
SRC=$HOME/src
BIN=$PREFIX/bin

# Étape 1 : Créer les dossiers
mkdir -p "$SRC"
mkdir -p "$BIN"
cd "$SRC"

# Étape 2 : Télécharger et extraire tmux
curl -LO https://github.com/tmux/tmux/releases/download/${TMUX_VERSION}/tmux-${TMUX_VERSION}.tar.gz
tar -xzf tmux-${TMUX_VERSION}.tar.gz
cd tmux-${TMUX_VERSION}

# Étape 3 : Télécharger les dépendances locales (si non présentes)
if [ ! -d "$SRC/libevent" ]; then
  curl -LO https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz
  tar -xzf libevent-2.1.12-stable.tar.gz
  cd libevent-2.1.12-stable
  ./configure --prefix=$PREFIX --disable-shared
  make -j4 && make install
  cd ..
fi

# Étape 4 : Compiler tmux
cd "$SRC/tmux-${TMUX_VERSION}"
./configure CFLAGS="-I$PREFIX/include" LDFLAGS="-L$PREFIX/lib" --prefix=$PREFIX
make -j4 && make install

# Étape 5 : Ajouter au PATH si nécessaire
if ! grep -q "$HOME/.local/bin" ~/.bashrc; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "✅ tmux installé localement. Recharge ton terminal ou fais : source ~/.bashrc"
