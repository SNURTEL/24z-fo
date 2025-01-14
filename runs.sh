python -m fo.scripts.train -a o3 -e 30 -t 2 ;
echo "run1";
python -m fo.scripts.train -a resnet18 -e 30 -t 2 ;
echo "run2";
python -m fo.scripts.train -a resnet18 -e 30 -t 2 -p;
echo "run3";

python -m fo.scripts.test -a o3 > results_o3.txt ;
echo "run4";
python -m fo.scripts.test -a resnet18 > results_resnet.txt ;
echo "run5";