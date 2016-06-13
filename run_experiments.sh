#!/bin/bash
clear
clear
for dir in ./test-data/DS*/
do
    dir=${dir%*/}
    echo "==================================================" 
    echo ${dir##*/}
    echo "==================================================" 
    python main.py evaluate ./test-data/${dir##*/}
done
