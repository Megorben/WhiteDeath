#!/usr/bin/env sh
# Downloads the Mindustry server jar and compiles the RL simulator.
set -e
cd "$(dirname "$0")"

VERSION=v146
mkdir -p lib build
if [ ! -f "lib/server-$VERSION.jar" ]; then
  echo "Downloading Mindustry server $VERSION..."
  curl -fL -o "lib/server-$VERSION.jar" "https://github.com/Anuken/Mindustry/releases/download/$VERSION/server-release.jar"
fi

javac -nowarn -d build -cp "lib/server-$VERSION.jar" $(find src -name '*.java')
echo "Built. Run with: java -cp build:lib/server-$VERSION.jar whitedeath.rl.SimServer"
