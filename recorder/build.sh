#!/usr/bin/env sh
# Builds dist/WhiteDeathRecorder.jar (desktop Mindustry build 160+). Pass --test to also run the headless check.
set -e
cd "$(dirname "$0")"

# Same build as the Steam release; Java mods must declare minGameVersion >= 154 or the game refuses them
VERSION=v160.7
JAR=../rl/sim/lib/server-$VERSION.jar
if [ ! -f "$JAR" ]; then
  mkdir -p ../rl/sim/lib
  curl -fL -o "$JAR" "https://github.com/Anuken/Mindustry/releases/download/$VERSION/server-release.jar"
fi

rm -rf build && mkdir -p build/classes dist
javac -nowarn --release 17 -d build/classes -cp "$JAR" $(find src -name '*.java')
cp mod.hjson build/classes/
cp -r bundles build/classes/
(cd build/classes && jar cf ../../dist/WhiteDeathRecorder.jar .)
echo "Built dist/WhiteDeathRecorder.jar"

if [ "$1" = "--test" ]; then
  javac -nowarn --release 17 -d build/test-classes -cp "$JAR:build/classes" $(find test -name '*.java')
  java -cp "build/test-classes:build/classes:$JAR" whitedeath.recorder.RecorderHarness build/test-output
fi
