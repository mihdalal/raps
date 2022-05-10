cd $1/rlkit
pip install -e .
mkdir -p data/
cd $1/d4rl
pip install -e .
cd $1/metaworld
pip install -e .
cd $1/robosuite
pip install -e .
cd $1/pytorch-a2c-ppo-acktr-gail/
pip install -e .
cd $1/rad/
pip install -e .
cd $1/viskit/
pip install -e .
cd $1/doodad/
pip install -e .
cd $1
pip install -r requirements.txt
